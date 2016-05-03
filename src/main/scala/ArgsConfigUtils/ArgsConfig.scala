package ArgsConfigUtils

/**
  * Author: Joojay Huyn joojayhuyn@utexas.edu
  * Created on 4/29/16
  * Utility functions to process command line arguments for deep learning and machine learning driver programs
  */
object ArgsConfig {

  // Constants used in keys of map
  val AppName = "appName"
  val DataDir = "dataDir"
  val MalFileName = "malFileName"
  val BenFileName = "benFileName"
  val SparkMaster = "sparkMaster"

  /*
   * Utility function to process command line arguments for deep learning and machine learning driver programs
   */
  def config(args: Array[String]): Map[String, String] = {
    // args is expected to be in the following format:
    // <app name> <data dir> <mal file> <ben file> <use aws flag> <optional master public dns>

    // Getting application name arg from args array
    val appName = args(0)

    // This implementation assumes 2 things: 1) Important data files, such as the malware traces data file and benign-
    // ware traces data file, are stored in the same data directory (data directory could lie in HDFS or file system of
    // Linux or MAC OS. 2) There are only 2 data files of interest.
    // In the future, if the problem concerns > 2 classes, then there will be > 2 files of interest. Hence, the code on
    // the next 3 lines is a hack and must be fixed to address future problems that involve > 2 classes.
    var dataDir = args(1)
    val malFileName = args(2)
    val benFileName = args(3)

    // 5th arg in array represents a boolean indicating whether the program will run on local machine or Spark AWS EC2
    // cluster
    val awsFlag = args(4).toBoolean

    // Initializing sparkMaster arg to an empty string
    var sparkMaster = ""

    // If program will run on AWS, make appropriate changes to sparkMaster and dataDir, which assumes that HDFS has
    // been setup on Spark AWS EC2 cluster
    if (awsFlag) {
      val masterPublicDNS = args(5)
      sparkMaster = "spark://" + masterPublicDNS + ":7077"
      val hdfs = "hdfs://" + masterPublicDNS + ":9000"
      dataDir = hdfs + dataDir
    } else { // else set sparkMaster to local
      sparkMaster = "local"
    }

    // Return map
    // Note that the 2 keys, MalFileName and BenFileName, suggest that there are only 2 data files (and hence 2 classes)
    // of interest for the current machine learning research problem. Obviously, this implementation is a hack and must
    // be fixed to address future problems concerning > 2 classes.
    Map(AppName -> appName, DataDir -> dataDir, MalFileName -> malFileName, BenFileName -> benFileName,
      SparkMaster -> sparkMaster)
  }

}
