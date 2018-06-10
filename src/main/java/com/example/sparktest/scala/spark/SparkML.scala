package com.example.sparktest.scala.spark

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LabeledPoint

object SparkML {

  def main(args: Array[String]): Unit = {
    KMeansAnalyzer
    //SVMAnalyzer
  }

  /**
    * ML K-Means
    */
  def KMeansAnalyzer: Unit = {

    val conf = new SparkConf().setAppName("KMeans")
    conf.setMaster("local")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    val data = sc.textFile("mllib\\kmeans_data.txt")
    val parsedData = data.map{
      line => {
        Vectors.dense(line.split(" ").map(_.toDouble))
      }
    }.cache()

    // 设置迭代次数、类簇的个数
    val maxIterations = 100
    val numIterations = 20
    val numClusters = 2
    var clusterIndex = 0

    // 进行训练
    val clusters = KMeans.train(parsedData, numClusters, maxIterations, numIterations)

    println("Cluster Centers Information Overview:")
    clusters.clusterCenters.foreach(x => {
      println("Center Point of Cluster " + clusterIndex + " : " + x)
      clusterIndex+=1
    })

    // 统计聚类错误的样本比例
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    //预测
    val predictVector = Vectors.dense(50.0)
    val index = clusters.predict(predictVector)
    println("Vector: " + predictVector + " belongs to cluster: " + index)
  }

  /**
    * ML SVM
    */
  def SVMAnalyzer: Unit = {

    val conf = new SparkConf().setAppName("KMeans")
    conf.setMaster("local")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    val data = sc.textFile("mllib\\svm_data.txt")
    //val data = sc.textFile("mllib\\sample_svm_data.txt")
    val parsedData = data.map{
      line => val parts = line.split(" ")
        //label, feature
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(_.toDouble)))
    }.cache()

    //迭代次数
    val numIterations = 200

    //开始训练
    //val model = SVMWithSGD.train(parsedData, numIterations)
    //model.clearThreshold()

    // 设定训练策略
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer.setNumIterations(numIterations)
      .setRegParam(0.1)
      .setUpdater(new L1Updater)
    val model = svmAlg.run(parsedData)

    // 统计分类错误的样本比例
    val labelAndPreds = parsedData.map{
      point => val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)

    //预测
    val predictVector = Vectors.dense(100.0, 100.0)
    val predictLabel = model.predict(predictVector)
    println("feature: " + predictVector + " belongs to label: " + predictLabel)
  }

}
