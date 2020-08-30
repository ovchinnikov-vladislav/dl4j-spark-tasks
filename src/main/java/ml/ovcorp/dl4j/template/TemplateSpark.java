package ml.ovcorp.dl4j.template;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

public class TemplateSpark {

    private static final Logger log = LoggerFactory.getLogger(ml.ovcorp.dl4j.template.TemplateSpark.class);

    public static void main(String[] args) throws Exception {
        new ml.ovcorp.dl4j.template.TemplateSpark().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        int batchSize = 16;
        int averagingFrequency = 1;
        int workerPrefetchNumBatches = 2;

        SparkConf sparkConf = new SparkConf();
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer"); // Сериализатор Kryo для Spark
        sparkConf.set("spark.kryo.registrator", "org.nd4j.kryo.Nd4jRegistrator");        // Регистратор Kryo для Nd4j
        //sparkConf.set("spark.hadoop.fs.defaultFS", "hdfs://192.168.0.12:9000");        // Адрес hadoop

        boolean useSparkLocal = true;                                                   // true - локальные вычисления, false - кластерные
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Spark MLP Template");                                 // Имя контекста Spark (имя приложения)
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<DataSet> train = getTrains(batchSize, sc);
        JavaRDD<DataSet> test = getTests(batchSize, sc);

        MultiLayerNetwork net = new MultiLayerNetwork(getMultiLayerConf());
        net.init();

        log.info("Количество параметров модели: {}", net.numParams());

        // Конфигурация для обучения на Spark
        TrainingMaster<?, ?> tm = new ParameterAveragingTrainingMaster
                .Builder(batchSize) // Каждый объект DataSet по умолчанию
                // Количество исполнителей на одной машине
                .averagingFrequency(averagingFrequency)
                .workerPrefetchNumBatches(workerPrefetchNumBatches) // Асинхронная предвыборка: 2 примера
                 // содержит 32 примера на каждого исполнителя
                .batchSizePerWorker(batchSize)
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .build();

        // Создать сеть Spark
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, net, tm);
        sparkNet.setListeners(Collections.singletonList(new ScoreIterationListener(1)));

        // Обучить сеть
        log.info("--- Начинается обучение сети ---");
        int nEpochs = 1;
        for (int i = 0; i < nEpochs; i++) {
            sparkNet.fit(train);
            log.info("----- Период " + i + " завершен -----");
            // Оценить с помощью Spark:
            Evaluation evaluation = sparkNet.evaluate(test);
            log.info(evaluation.stats());
        }
        log.info("****************Конец примера********************");
    }

    // Пример получения JavaRDD для обучения с использованием Spark (набор данных Mnist)
    private static JavaRDD<DataSet> getTrains(int batchSize, JavaSparkContext sc) throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        List<DataSet> trainData = new ArrayList<>();
        while (mnistTrain.hasNext()) {
            trainData.add(mnistTrain.next());
        }
        Collections.shuffle(trainData, new Random(12345));

        // Получить обучающие данные. В реальных задачах
        // использовать parallelize не рекомендуется
        return sc.parallelize(trainData);
    }

    // Пример получения JavaRDD для обучения с использованием Spark (набор данных Mnist)
    private static JavaRDD<DataSet> getTests(int batchSize, JavaSparkContext sc) throws IOException {
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

        List<DataSet> testData = new ArrayList<>();
        while (mnistTest.hasNext()) {
            testData.add(mnistTest.next());
        }

        // Получить тестовые данные. В реальных задачах
        // использовать parallelize не рекомендуется
        return sc.parallelize(testData);
    }

    // Сконфигурировать сеть (как стандартную сеть в DL4J)
    private static MultiLayerConfiguration getMultiLayerConf() {
        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        learningRateSchedule.put(0, 0.06);
        learningRateSchedule.put(200, 0.05);
        learningRateSchedule.put(600, 0.028);
        learningRateSchedule.put(800, 0.0060);
        learningRateSchedule.put(1000, 0.001);

        int nChannels = 1;
        int outputNum = 10;
        int seed = 123;
        int width = 28;
        int height = 28;

        log.info("Построение модели....");
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, nChannels)) // InputType.convolutional for normal image
                .build();
    }
}
