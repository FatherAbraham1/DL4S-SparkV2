package MBartlingDL4J

// Importing necessary libraries
import ArgsConfigUtils.ArgsConfig
import ETLUtils.ETL
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Author: Joojay Huyn joojayhuyn@utexas.edu
  * Created on 4/14/16.
  * This driver program contains main a function to load a dataset and train and run an mlp model on a test set to
  * measure the model's performance.
  */
object MLPDriver {

  /*
   * Main function creates a Spark RDD of Labeled Points from csv data files. Then, an MLP model is trained and
   * ran through a final unseen test set to return its true performance metric.
   */
  def main(args: Array[String]) {
    // args is expected to be in the following format:
    // <app name> <data dir> <mal file> <ben file> <use aws flag> <optional master public dns>

    // Configuring args by processing an array of command line arguments and returning a map of values
    val argsMap = ArgsConfig.config(args)
    val appName = argsMap(ArgsConfig.AppName)
    val dataDir = argsMap(ArgsConfig.DataDir)
    val malFileName = argsMap(ArgsConfig.MalFileName)
    val benFileName = argsMap(ArgsConfig.BenFileName)
    val sparkMaster = argsMap(ArgsConfig.SparkMaster)

    // Creating a Spark context
    val conf = new SparkConf().setMaster(sparkMaster).setAppName(appName)
    val sc = new SparkContext(conf)

    // Initializing data file names
    // This is a hack because it assumes that there are only 2 data files of interest and must be fixed to address
    // future problems concerning > 2 data files on interest. See notes in ArgsConfigUtils.ArgsConfig scala object.
    val malDataFileName = dataDir + malFileName
    val benDataFileName = dataDir + benFileName

    // Converting csv data files (one file for each class, malicious and benign) to a Spark RDD of Labeled Points.
    // Once again, this function is a hack because it assumes there are only 2 classes of interest.
    val rddLP = ETL.mBartCSVToRDDLP(sc, malDataFileName, benDataFileName)

    // Getting training and test dataset from Spark RDD of interest
    val (trainingData, testData) = ETL.getTrainTestSetsFromRDDLP(rddLP)

    // Caching training dataset to accelerate ML model training
    trainingData.cache()

    println("Size of entire training dataset: " + trainingData.count())

    // Configuring multilayer perceptron
    val numInputs = 332
    val outputNum = 2
    val iterations = 100

    val nnConf = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .iterations(100)
      .optimizationAlgo(OptimizationAlgorithm.LBFGS)
      .learningRate(1.0)
      .l2(0.01).regularization(true)
      .list(2)
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(150)
        .activation("sigmoid")
        .weightInit(WeightInit.NORMALIZED)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(150).nOut(outputNum)
        .activation("softmax")
        .weightInit(WeightInit.NORMALIZED)
        .build())
      .backprop(true).pretrain(false)
      .build()

    val net  = new MultiLayerNetwork(nnConf)
    net.init()
    net.setUpdater(null)

    // Create Spark Network model
    val model = new SparkDl4jMultiLayer(sc, net)

    // Training mlp model
    model.fit(sc, trainingData)

    // Evaluating performance of lr model on never-before-seen test set and obtaining area under roc performance metric
    println("Size of entire testing dataset: " + testData.count())
    val testPredictionAndLabels = testData.map { point =>
      val rawPrediction = model.predict(point.features)
      val prediction = rawPrediction.argmax.toDouble
      (prediction, point.label)
    }
    val testMetrics = new BinaryClassificationMetrics(testPredictionAndLabels)
    println("Final testing areaUnderROC: " + testMetrics.areaUnderROC())

    // Obtaining accuracy rate of SVM model on test set
    val numMisPredicted = testData.filter(point =>
      model.predict(point.features).argmax.toDouble != point.label).count().toDouble
    val testErr = numMisPredicted / testData.count()
    val accRate = 1.0 - testErr
    println("Final testing accuracy rate: " + accRate)

  }
}
