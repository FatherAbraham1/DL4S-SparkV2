package MBartlingSparkML

// Importing necessary libraries
import ArgsConfigUtils.ArgsConfig
import ETLUtils.ETL
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

/**
  * Author: Joojay Huyn joojayhuyn@utexas.edu
  * Created on 4/12/16
  * This driver program contains main a function to load a dataset and train and run a logistic regression model on
  * a test set to measure the model's performance.
  */
object LogRegDriver {

  /*
   * Main function creates a Spark DataFrame from csv data files. Then, a logistic regression model is trained and
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

    // Creating a SQL Context
    val sqlContext = new SQLContext(sc)

    // Converting csv data files (one file for each class, malicious and benign) to a Spark DataFrame. Once again,
    // this function is a hack because it assumes there are only 2 classes of interest.
    //val (malDataFrame, benDataFrame) = ETL.mBartCSVToMLDataframe(sc, sqlContext, malDataFileName, benDataFileName)
    val df = ETL.mBartCSVToMLDataFrame(sc, sqlContext, malDataFileName, benDataFileName)

    // Getting training and test dataset from Spark DataFrame of interest
    val (trainingData, testData) = ETL.getTrainTestSetsFromDataFrame(df)

    // Caching the trainingData DataFrame will accelerate model training
    trainingData.cache()

    println("Size of entire training dataset: " + trainingData.count())

    // Configuring logistic regression (lr) model with default parameter values
    val lr = new LogisticRegression()

    // Training lr model
    val lrModel = lr.fit(trainingData)

    // Evaluating performance of lr model on never-before-seen test set and obtaining area under roc performance metric
    println("Size of entire testing dataset: " + testData.count())
    val testResult = lrModel.transform(testData)
    val testRawPredictionAndLabelsDataFrame = testResult.select("rawPrediction", "label")
    val evaluator = new BinaryClassificationEvaluator() // default value of evaluator is areaUnderROC
    println("Final testing areaUnderROC: " + evaluator.evaluate(testRawPredictionAndLabelsDataFrame))

    // Obtaining accuracy rate of lr model on test set
    testResult.registerTempTable("mBartData")
    val numMisPredicted = sqlContext.sql("SELECT COUNT(*) FROM mBartData WHERE prediction <> label")
      .first()(0).asInstanceOf[Long].toDouble
    val testErr = numMisPredicted / testData.count()
    val accRate = 1.0 - testErr
    println("Final testing accuracy rate: " + accRate)

  }
}
