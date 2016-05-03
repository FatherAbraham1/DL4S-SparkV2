package MBartlingSparkML

// Importing necessary libraries
import ArgsConfigUtils.ArgsConfig
import ETLUtils.ETL
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Author: Joojay Huyn joojayhuyn@utexas.edu
  * Created on 4/18/16
  * This driver program contains main a function to load a dataset and train and run an SVM model on a test set to
  * measure the model's  performance.
  */
object SVMDriver {

  /*
   * Main function creates a Spark RDD of Labeled Points from csv data files. Then, an SVM model is trained and
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

    // Caching the trainingData RDD will accelerate model training
    trainingData.cache()

    println("Size of entire training dataset: " + trainingData.count())

    // Configuring SVM model with default parameter values and training the model
    val numIterations = 100
    val model = SVMWithSGD.train(trainingData, numIterations)

    // Evaluating performance of svm model on never-before-seen test set and obtaining area under roc performance metric
    println("Size of entire testing dataset: " + testData.count())
    val testPredictionAndLabels = testData.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }
    val testMetrics = new BinaryClassificationMetrics(testPredictionAndLabels)
    println("Final testing areaUnderROC: " + testMetrics.areaUnderROC())

    // Obtaining accuracy rate of SVM model on test set
    val numMisPredicted = testData.filter(point => model.predict(point.features) != point.label).count().toDouble
    val testErr = numMisPredicted / testData.count()
    val accRate = 1.0 - testErr
    println("Final testing accuracy rate: " + accRate)

  }

}
