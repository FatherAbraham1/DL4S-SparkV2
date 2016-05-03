name := "DL4S-SparkV2"

version := "1.0"

scalaVersion := "2.10.6"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.apache.spark" %% "spark-mllib" % "1.6.0",
  "org.nd4j" % "nd4j-x86" % "0.4-rc3.8",
  "org.nd4j" % "canova-api" % "0.0.0.14",
  "org.nd4j" % "nd4j-jblas" % "0.4-rc3.6",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8",
  "org.deeplearning4j" % "deeplearning4j-cli" % "0.4-rc3.8",
  "org.deeplearning4j" % "deeplearning4j-scaleout" % "0.4-rc3.8",
  "org.deeplearning4j" % "dl4j-spark" % "0.4-rc3.8"
)