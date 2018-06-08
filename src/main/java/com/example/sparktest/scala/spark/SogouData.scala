package com.example.sparktest.scala.spark

import org.apache.spark.sql.{Row, SparkSession}

case class SogouRow(time: String, id: String, des: String, number: Int, url: String)

object SogouData {

  def main(args: Array[String]): Unit = {
    sogouFileAnalysis
  }

  def sogouFileAnalysis: java.util.List[_] = {

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Sogou Spark Sql")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    //Read Sougou File into RDD
    val fileData = spark.sparkContext.textFile("data\\SogouQ.reduced\\SogouQ.reduced")
    //Map each row into a case class
    val splitRDD = fileData.flatMap(file => file.split("\n")).map(line => line.split("\t"))
      .map(attributes => SogouRow(attributes(0), attributes(1), attributes(2), attributes(3).replace(" ","").toInt, attributes(4)))

    //create dataFrame for RDD
    val fileDataDF = spark.createDataFrame(splitRDD)
    fileDataDF.createOrReplaceTempView("sougouQ")
    spark.sql("select * from sougouQ").show()

    //transform dataFrame to RDD
    val list = spark.sql("select * from sougouQ").rdd.map{
      case Row(time: String, id: String, des: String, number: Int, url: String) => SogouRow(time, id, des, number, url)
    }.toJavaRDD().collect()

    println(list.size())
    return list
  }

}
