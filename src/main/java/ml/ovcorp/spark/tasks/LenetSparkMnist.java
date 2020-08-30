package ml.ovcorp.spark.tasks;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;

public class LenetSparkMnist {

    private static final Logger log = LoggerFactory.getLogger(LenetSparkMnist.class);

    public static void main(String[] args) throws Exception {
        new LenetSparkMnist().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        int batchSize = 100;
        int averagingFrequency = 6;
        int workerPrefetchNumBatches = 2;
        int seed = 12345;

        SparkConf sparkConf = new SparkConf();
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer"); // Сериализатор Kryo для Spark
        sparkConf.set("spark.kryo.registrator", "org.nd4j.kryo.Nd4jRegistrator");        // Регистратор Kryo для Nd4j
        //sparkConf.set("spark.hadoop.fs.defaultFS", "hdfs://192.168.0.12:9000");        // Адрес hadoop

        boolean useSparkLocal = false;                                                   // true - локальные вычисления, false - кластерные
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Spark Lenet Mnist");                                 // Имя контекста Spark (имя приложения)
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<DataSet> train = getTrains(batchSize, sc);
        JavaRDD<DataSet> test = getTests(batchSize, sc);

        log.info("Построение модели...");
        ComputationGraph net = new ComputationGraph(getLenetConf(seed));
        net.init();

        log.info("Конфигурация модели:\n{}", net.getConfiguration().toJson());
        log.info("Количество параметров модели: {}", net.numParams());
        log.info("Анализ информации о слоях.");
        int i = 0;
        for (Layer l : net.getLayers()) {
            log.info("{}. Тип слоя: {}. Количество параметров слоя: {}.",
                    ++i,
                    l.type(),
                    l.numParams());
        }

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
        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, tm);
        sparkNet.setListeners(Collections.singletonList(new ScoreIterationListener(1)));

        // Обучить сеть
        log.info("--- Начинается обучение сети ---");
        LocalDateTime start = LocalDateTime.now();
        log.info("Время начала обучения: {}", start);
        int nEpochs = 1;
        for (i = 0; i < nEpochs; i++) {
            sparkNet.fit(train);
            log.info("----- Период " + i + " завершен -----");
            // Оценить с помощью Spark:
            Evaluation evaluation = sparkNet.evaluate(test);
            log.info(evaluation.stats());
        }

        LocalDateTime end = LocalDateTime.now();
        log.info("Размер пакета: {}", batchSize);
        log.info("Время начала обучения: {}", start);
        log.info("Время конца обучения: {}", end);
        log.info("Количество эпох: {}", nEpochs);
        diffLocalDateTime(start, end);

        log.info("****************Конец обучения********************");
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

    private static ComputationGraphConfiguration getLenetConf(int seed) {
        int nChannels = 1;
        int nClasses = 10;
        int height = 28;
        int width = 28;

        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutionalFlat(height, width, nChannels))
                .addLayer("cnn1",
                        new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                                .name("cnn1")
                                .nIn(nChannels)
                                .nOut(50)
                                .biasInit(0)
                                .build(), "input")
                .addLayer("maxpool1",
                        new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                                .name("maxpool1")
                                .build(), "cnn1")
                .addLayer("cnn2",
                        new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{5, 5}, new int[]{1, 1})
                                .name("cnn2")
                                .nOut(100)
                                .biasInit(0)
                                .build(), "maxpool1")
                .addLayer("maxpool2",
                        new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                                .name("maxool2")
                                .build(), "cnn2")
                .addLayer("denseLayer",
                        new DenseLayer.Builder()
                                .nOut(500)
                                .build(), "maxpool2")
                .addLayer("outputLayer",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(nClasses)
                                .activation(Activation.SOFTMAX)
                                .build(), "denseLayer")
                .setOutputs("outputLayer")
                .build();
    }

    private static void diffLocalDateTime(LocalDateTime from, LocalDateTime to) {

        LocalDateTime tempDateTime = LocalDateTime.from(from);

        StringBuilder timeString = new StringBuilder();

        long years = tempDateTime.until(to, ChronoUnit.YEARS);
        tempDateTime = tempDateTime.plusYears(years);

        if (years > 0) {
            timeString.append(years).append(" y ");
        }

        long months = tempDateTime.until(to, ChronoUnit.MONTHS);
        tempDateTime = tempDateTime.plusMonths(months);

        if (months > 0) {
            timeString.append(months).append(" mn ");
        }

        long days = tempDateTime.until(to, ChronoUnit.DAYS);
        tempDateTime = tempDateTime.plusDays(days);

        if (days > 0) {
            timeString.append(days).append(" d ");
        }

        long hours = tempDateTime.until(to, ChronoUnit.HOURS);
        tempDateTime = tempDateTime.plusHours(hours);

        if (hours > 0) {
            timeString.append(hours).append(" h ");
        }

        long minutes = tempDateTime.until(to, ChronoUnit.MINUTES);
        tempDateTime = tempDateTime.plusMinutes(minutes);

        if (minutes > 0) {
            timeString.append(minutes).append(" m ");
        }

        long seconds = tempDateTime.until(to, ChronoUnit.SECONDS);
        tempDateTime = tempDateTime.plusSeconds(seconds);

        if (seconds > 0) {
            timeString.append(seconds).append(" s ");
        }

        long milliseconds = tempDateTime.until(to, ChronoUnit.MILLIS);

        if (milliseconds > 0) {
            timeString.append(milliseconds).append(" ms ");
        }

        log.info("Общее время обучения составило: {}", timeString);
    }
}
