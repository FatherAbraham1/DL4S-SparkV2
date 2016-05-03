package MBartlingSparkML

// Importing necessary libraries
import ArgsConfigUtils.ArgsConfig
import ETLUtils.ETL
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Author: Joojay Huyn joojayhuyn@utexas.edu
  * Created on 4/18/16
  * This driver program contains main a function to load a dataset and run 10-fold cross validation on the dataset with
  * a different multilayer perceptron (mlp) models. Then, the best selected model runs predictions on a final test set
  * to measure the model's true performance.
  */
object MLPDriver {

  /*
   * Main function creates a Spark DataFrame from csv data files. Then, it runs 10-fold cross validation on several
   * different architectures for mlp models. After the best architecture for the mlp model is selected, the model
   * runs through a final unseen testset to return its true performance metric.
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

    // Splitting training dataset for 10-fold cross validation
    val splits = trainingData.randomSplit(Array(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), seed = 1234L)

    // MLP architectures to be tested
    val layersArray = Array(Array[Int](332, 150, 2), Array[Int](332, 200, 100, 2), Array[Int](332, 150, 150, 2))

    // Initializing best layers architecture and best avg area under roc to empty and 0 for 10-fold cross validation
    // respectively
    var bestLayers = Array[Int]()
    var bestAvgAreaUnderROC = 0.0

    /*
     * Executing 10-fold cross validation on MLP architectures of interest and selecting architecture with best
     * avgAreaUnderROC. Note that the for comprehension code below is ugly. Unfortunately, I cannot take advantage
     * of the spark.ml cross validation pipeline because: 1) the spark.ml.evaluation.MulticlassClassificationEvaluator
     * class does not support the areaUnderROC metric 2) the
     * spark.ml.classification.MultilayerPerceptronClassificationModel class and the
     * spark.ml.evaluation.BinaryClassificationEvaluator are incompatible - conflicts began to occur in the schema of
     * the generated Spark DataFrame.
     */
    for (i <- 0 until layersArray.length) { // Iterating through each MLP architecture

      // Getting MLP architecture and initializing avgAreaUnderRoc to 0
      val layers = layersArray(i)
      var avgAreaUnderROC = 0.0

      // Iterating through each fold in 10 fold cross validation
      for (j <- 0 until splits.length) {

        // Getting cross validation set
        val cvSet = splits(j)

        // Getting training set, which consists of the remaining 9 folds that are not assigned to the cross validation
        // set
        val trainSetBuffer = splits.toBuffer
        trainSetBuffer.remove(j)
        val trainSet = trainSetBuffer.toList.reduce((df1, df2) => df1.unionAll(df2))
        println("Iteration: " + j + " Size of training dataset: " + trainSet.count() + " Size of cv dataset: " +
          cvSet.count() + " MLP Architecture: " + layers.mkString(","))

        // Configuring multilayer perceptron with current architecture and other default parameter values
        val trainingMLP = new MultilayerPerceptronClassifier()
          .setLayers(layers)
          .setBlockSize(128)
          .setSeed(1234L)
          .setMaxIter(100)

        // Training mlp model
        val trainingMLPModel = trainingMLP.fit(trainSet)

        // Evaluating performance of mlp model on cross validation set
        val result = trainingMLPModel.transform(cvSet)

        // this DataFrame schema has 2 columns, prediction and label
        val predictionAndLabelsDataFrame = result.select("prediction", "label")

        // Converting DataFrame to RDD. Unfortunately, there does not seem to be a way around this hack currently as
        // explain above.
        val predictionAndLabels = predictionAndLabelsDataFrame.rdd.map(row =>
          (row(0).asInstanceOf[Double], row(1).asInstanceOf[Double]))
        val metrics = new BinaryClassificationMetrics(predictionAndLabels)

        // Add to avg area under roc metric
        avgAreaUnderROC += metrics.areaUnderROC()
      }

      avgAreaUnderROC /= 10 // Dividing avg area under roc by 10 to get the correct avg
      println("Training average areaUnderROC: " + avgAreaUnderROC)

      // Updating best avg area under roc and best architecture seen so far
      if (avgAreaUnderROC > bestAvgAreaUnderROC) {
        bestAvgAreaUnderROC = avgAreaUnderROC
        bestLayers = layers
      }
    }

    //bestLayers = Array[Int](332, 150, 2)
    println("Selected MLP's architecture: " + bestLayers.mkString(","))

    // Configuring multilayer perceptron with selected architecture and other default parameter values
    val mlp = new MultilayerPerceptronClassifier()
      .setLayers(bestLayers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // Training mlp model
    val model = mlp.fit(trainingData)

    // Evaluating performance of mlp model on never-before-seen test set and obtaining area under roc performance metric
    println("Size of entire testing dataset: " + testData.count())
    val testResult = model.transform(testData)
    val testPredictionAndLabelsDataFrame = testResult.select("prediction", "label")
    val testPredictionAndLabels = testPredictionAndLabelsDataFrame.rdd.map(row =>
      (row(0).asInstanceOf[Double], row(1).asInstanceOf[Double]))
    val testMetrics = new BinaryClassificationMetrics(testPredictionAndLabels)
    println("Final testing areaUnderROC: " + testMetrics.areaUnderROC())

    // Obtaining accuracy rate of MLP model on test set
    testResult.registerTempTable("mBartData")
    val numMisPredicted = sqlContext.sql("SELECT COUNT(*) FROM mBartData WHERE prediction <> label")
      .first()(0).asInstanceOf[Long].toDouble
    val testErr = numMisPredicted / testData.count()
    val accRate = 1.0 - testErr
    println("Final testing accuracy rate: " + accRate)
  }
}
