package ETLUtils

// Importing necessary libraries
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._

import scala.io.Source

/**
  * Author: Joojay Huyn joojayhuyn@utexas.edu
  * Created on 4/18/16
  * Object contains utility functions to perform extraction, transformation, and loading operations on csv file data
  * to a Spark collection (DataFrame or RDD)
  */
object ETL {

  // Symbolic Constants representing numerical values of class labels
  val MalClass = 1.0
  val BenClass = 0.0

  /*
   * Function that converts csv files of Michael Bartling's data to a pair of RDDs of vectors. Each element of the pair
   * corresponds to an RDD representing the malware data or benignware data. This function is a hack because it
   * assumes there there are 2 data files of interest and 2 classes of interest in the machine learning problem, hence
   * the return type of a pair of RDDs. This function should be changed to address future problems that concern > 2
   * data files of interest and > 2 classes of interest.
   */
  def mBartCSVToPairRDDVec(sc: SparkContext, malDataFileName: String, benDataFileName: String):
    (RDD[Vector], RDD[Vector]) = {

    /*
     * Getting malware and benignware datasets (Mal.csv and Ben.csv). Each row in a csv represents one data instance.
     * Because each row has 332 columns, each data instance is a vector of 332 dimensions or features. Each of the 332
     * dimensions corresponds to one of the 332 instructions. For example, a numeric value in the 2nd column and 3rd row
     * represents the number of times <instruction 2> was called during the running of that data instance 3. Recall
     * that when an instance runs, it executes a set of instructions.
     */
    val malInput = sc.textFile(malDataFileName)
    val benInput = sc.textFile(benDataFileName)

    // Transforming each line of each csv file into a vector of 332 dimensions to obtain a dataset of vectors for each
    // csv file.
    val malRawVectors = malInput.map(line => line.split(",").map(str => str.toDouble)).map( arr => Vectors.dense(arr) )
    val benRawVectors = benInput.map(line => line.split(",").map(str => str.toDouble)).map( arr => Vectors.dense(arr) )

    // Creating super raw vector which is the union of all the datasets
    val allRawVectors = malRawVectors.union(benRawVectors)

    // Normalizing and scaling vectors
    val scaler = new StandardScaler().fit(allRawVectors)
    val malScaledVectors = scaler.transform(malRawVectors)
    val benScaledVectors = scaler.transform(benRawVectors)

    // Returning pair of RDD of scaled vectors
    (malScaledVectors, benScaledVectors)
  }

  /*
   * Function that takes data files of interest, converts that into RDDs of LabeledPoints, and combines the RDDs to
   * return one large RDD of LabeledPoints. Once again, this function is a hack because it assumes that there are only
   * 2 data files of interest. This function must be changed in the future to address situations where there are > 2
   * data files of interest.
   */
  def mBartCSVToRDDLP(sc: SparkContext, malDataFileName: String, benDataFileName: String): RDD[LabeledPoint] = {

    // Converting data files of interest to RDDs of scaled vectors. Each RDD corresponds to one file.
    val (malScaledVectors, benScaledVectors) = mBartCSVToPairRDDVec(sc, malDataFileName, benDataFileName)

    // Converting each RDD of vectors to an RDD of labeled points and combining everything into one RDD of labeled
    // points
    val malLabeledPointsRDD = malScaledVectors.map(featureVec => LabeledPoint(MalClass, featureVec))
    val benLabeledPointsRDD = benScaledVectors.map(featureVec => LabeledPoint(BenClass, featureVec))
    val labeledPointsRDD = malLabeledPointsRDD.union(benLabeledPointsRDD)

    // Returning RDD of labeled points
    labeledPointsRDD
  }

  /*
   * Function that takes data files of interest, converts them into DataFrames, combines the DataFrames to return one
   * large DataFrame. Once again, this function is a hack because it assumes that there are only 2 data files of
   * interest. This function must be changed in the future to address situations where there are > 2 data files of
   * interest.
   */
  def mBartCSVToMLDataFrame(sc: SparkContext, sqlContext: SQLContext, malDataFileName: String, benDataFileName: String):
    DataFrame = {

    // Converting data files of interest to RDDs of scaled vectors. Each RDD corresponds to one file.
    val (malScaledVectors, benScaledVectors) = mBartCSVToPairRDDVec(sc, malDataFileName, benDataFileName)

    // Converting each RDD of vectors to an RDD of rows and combining everything into one RDD of rows
    val malRowRDD = malScaledVectors.map(vector => Row(MalClass, vector))
    val benRowRDD = benScaledVectors.map(vector => Row(BenClass, vector))
    val rowRDD = malRowRDD.union(benRowRDD)

    // Creating schema for ML classifiers. This schema has 2 columns, a label (class) column of type double and a
    // features column of type vector
    val schema = StructType(
      StructField("label", DoubleType, true) ::
      StructField("features", new VectorUDT, true) :: Nil
    )

    // Creating DataFrame from RDD of rows and schema
    val df = sqlContext.createDataFrame(rowRDD, schema)

    // Returning DataFrame
    df
  }

  /*
   * Function returns a schema of type StructType (for DataFrame purposes) of Michael Bartling's data. The expected
   * schema has 332 columns, where each column refers to a specific system call. The value at row i and column j
   * represents the number of times system call j occurred in data instance i.
   */
  def getMBartCSVSchema(sc: SparkContext, malDataFileName: String, benDataFileName: String, schemaFileName: String):
  StructType = {

    // Get list of lines from schema file, where each line contains a distinct system call
    val fileLines = Source.fromFile(schemaFileName).getLines().toList

    // Create schema by adding a list of StructFields to a StructType.
    // Each StructField represents a distinct system call except for the last StructField, which represents the class
    // of that data instance/entire row.
    val schema = StructType( fileLines.map { fileLine =>
      val fieldName = fileLine.split(" ")(0)
      StructField(fieldName, DoubleType, true)
    }
    ).add(StructField("class", DoubleType, true))

    // Return schema
    schema
  }

  /*
   * Function takes a Spark RDD of Labeled Points and randomly assigns 80% of the RDD to a training set and assigns
   * the remaining 20% to the test set.
   */
  def getTrainTestSetsFromRDDLP(labeledPointsRDD: RDD[LabeledPoint]): (RDD[LabeledPoint], RDD[LabeledPoint]) = {

    // Randomly splitting initial RDD of labeled points into 2 RDDs of labeled points (training set and test set)
    // via an 80/20 split
    val splits = labeledPointsRDD.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val trainingRDD = splits(0)
    val testRDD = splits(1)

    // Returning training and test set
    (trainingRDD, testRDD)
  }

  /*
   * Function takes a Spark RDD of labeled points and randomly splits the RDD into 3 sets: training set (60%),
   * cross-validation set (20%), and test set (20%).
   */
  def getTrainCVTestSetsFromRDDLP(labeledPointsRDD: RDD[LabeledPoint]):
    (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {

    // Randomly splitting initial RDD of labeled points into 3 RDDs of labeled points (training, cross-validation,
    // test set)
    // via a 60/20/20 split
    val splits = labeledPointsRDD.randomSplit(Array(0.6, 0.2, 0.2), seed = 1234L)
    val trainingRDD = splits(0)
    val cvRDD = splits(1)
    val testRDD = splits(2)

    // Returning training, cv, and test set
    (trainingRDD, cvRDD, testRDD)
  }

  /*
   * Function takes a Spark DataFrame and randomly assigns 80% of the DataFrame's rows to a training set and assigns
   * the remaining 20% to the test set.
   */
  def getTrainTestSetsFromDataFrame(df: DataFrame): (DataFrame, DataFrame) = {

    // Randomly splitting initial DataFrame into 2 DataFrames (training set and test set) via an 80/20 split
    val splits = df.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val trainingDataFrame = splits(0)
    val testDataFrame = splits(1)

    // Return training and test set
    (trainingDataFrame, testDataFrame)
  }

  /*
   * Function takes a Spark DataFrame and randomly splits the DataFrame's rows into 3 sets: training set (60%),
   * cross-validation set (20%), and test set (20%).
   */
  def getTrainCVTestSetsFromDataFrame(df: DataFrame): (DataFrame, DataFrame, DataFrame) = {

    // Randomly splitting initial DataFrame into 3 DataFrames (training, cross-validation, test set)
    // via a 60/20/20 split
    val splits = df.randomSplit(Array(0.6, 0.2, 0.2), seed = 1234L)
    val trainingDataFrame = splits(0)
    val cvDataFrame = splits(1)
    val testDataFrame = splits(2)

    // Return training, cv, and test set
    (trainingDataFrame, cvDataFrame, testDataFrame)
  }

}
