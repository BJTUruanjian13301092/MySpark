package com.example.sparktest.scala.spark

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD

object SparkML {

  def main(args: Array[String]): Unit = {
    //KMeansAnalyzer
    //SVMAnalyzer
    //LinearRegressionAnalyzer
    //ALSRecommendationAnalyzer
    ALSMovieRecommendation
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

    sc.stop()
  }

  /**
    * ML SVM
    */
  def SVMAnalyzer: Unit = {

    val conf = new SparkConf().setAppName("SVM")
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
    val predictVector = Vectors.dense(1.0, 100.0)
    val predictLabel = model.predict(predictVector)
    println("feature: " + predictVector + " belongs to label: " + predictLabel)

    sc.stop()
  }

  /**
    * ML LinearRegression
    */
  def LinearRegressionAnalyzer: Unit = {

    val conf = new SparkConf().setAppName("LinearRegression")
    conf.setMaster("local")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    val data = sc.textFile("mllib\\lpsa.data")
    val parsedData = data.map {
      line => val parts = line.split(",")
        //label, feature
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(x => x.toDouble)))
    }.cache()

    //设置迭代次数并进行训练
    val numIterations = 20
    val model = LinearRegressionWithSGD.train(parsedData, numIterations)

    //统计回归错误的样本比例
    val valuesAndPreds = parsedData.map{
      point => val prediction = model.predict(point.features)
        (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.reduce(_ + _)/valuesAndPreds.count
    println("training Mean Squared Error = " + MSE)

    sc.stop()
  }

  /**
    * ML ALS
    */
  def ALSRecommendationAnalyzer: Unit = {

    val conf = new SparkConf().setAppName("ALS")
    conf.setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint")

    // 加载和解析数据文件
    val data = sc.textFile("mllib\\als\\test.data")
    val ratings = data.map(_.split(',') match {
      case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble)
    }).cache()

    // 设置迭代次数和Rank
    val numIterations = 20
    val rank = 10
    val model = ALS.train(ratings, rank, numIterations, 0.01)

    // 对推荐模型进行评分
    val usersProducts = ratings.map{ case Rating(user, product, rate) => (user, product)}
    val predictions = model.predict(usersProducts).map{
      case Rating(user, product, rate) => ((user, product), rate)
    }

    val ratesAndPreds = ratings.map{
      case Rating(user, product, rate) => ((user, product), rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map{
      case ((user, product), (r1, r2)) => math.pow((r1- r2), 2)
    }.reduce(_ + _)/ratesAndPreds.count

    println("Mean Squared Error = " + MSE)

    sc.stop()
  }

  /**
    * ML ALS for Movie Data
    */
  def ALSMovieRecommendation: Unit = {

    val conf = new SparkConf().setAppName("ALS for movie")
    conf.setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("spark-warehouse\\checkpoint")

    // 加载和解析数据文件

    // 加载打分
    var dataForRating = sc.textFile("mllib\\als\\ml-latest-small\\ml-latest-small\\ratings.csv")
    val headerForRating = dataForRating.first()
    dataForRating = dataForRating.filter(row => row != headerForRating)

    // 加载电影信息
    var dataForMovies = sc.textFile("mllib\\als\\ml-latest-small\\ml-latest-small\\movies.csv")
    val headerForMovies = dataForMovies.first()
    dataForMovies = dataForMovies.filter(row => row != headerForMovies)
    val movieRDD = dataForMovies.map{
      line => val fields = line.split(",")
        (fields(0).toInt, (fields(1), fields(2)))
    }.cache()

    //切分打分数据为训练数据和测试数据
    val ratingRDD = dataForRating.map{
      line => val fields = line.split(",")
        (fields(3).toLong, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = ratingRDD(0).cache()
    val testing = ratingRDD(1).cache()

    // 设置迭代次数和Rank
    val numIterations = 50
    val rank = 10
    val alpha = 0.01
    val model = ALS.train(training.values, rank, numIterations, alpha)

    //使用测试数据对模型进行评估
    val userProduct = testing.values.map{
      case Rating(user, product, rate) => (user, product)
    }
    val prediction = model.predict(userProduct).map{
      case Rating(user, product, rate) => ((user, product), rate)
    }
    val ratesAndPreds = testing.values.map{
      case Rating(user, product, rate) => ((user, product), rate)
    }.join(prediction)

    val MSE = ratesAndPreds.map{
      case ((user, product), (r1, r2)) => math.pow((r1- r2), 2)
    }.reduce(_ + _)/ratesAndPreds.count
    println("Mean Squared Error = " + MSE)
    println()

    // 预测用户最想看的前十部电影
    recommendMovieForUser(1, model, movieRDD, dataForRating)

//    // 组装（user product）
//    val userProductFor_1 = movieRDD.map(
//      // user, product
//      data => (1, data._1)
//    )
//    // 找出用户已经打过分的电影
//    val movieRated = dataForRating.map{
//      line => val fields = line.split(",")
//        (fields(0).toInt, fields(1).toInt)
//    }.filter(_._1 == 1)
//
//    val predictionForUser_1 = model.predict(userProductFor_1).map{
//      case Rating(user, product, rate) => (product, (user, rate))
//    }.join(movieRDD).sortBy(-_._2._1._2).take(10)
//
//    predictionForUser_1.foreach(
//      data =>
//      println("User: " + data._2._1._1 + " MovieId: " + data._1 + " MovieName: " + data._2._2._1 + " MovieType: " + data._2._2._2 + " Rate: " + data._2._1._2 )
//    )
//
//    println("--------------数据比对--------------")
//
//    val userProductFor_test = movieRated.map(
//      // user, product
//      data => (1, data._2)
//    )
//    val predictionForUser_test = model.predict(userProductFor_test).map{
//      case Rating(user, product, rate) => (user, product, rate)
//    }.foreach{
//      data => println("User: " + data._1 + " MovieId: " + data._2 + " Rate: " + data._3)
//    }

  }

  /**
    * 用户电影推荐
    * @param userId
    */
  def recommendMovieForUser(userId : Int, model : MatrixFactorizationModel,
                            movieRDD : RDD[(Int, (String, String))], dataForRating : RDD[String]): Unit = {
    // 组装（user product）
    val userProductFor_1 = movieRDD.map(
      // user, product
      data => (userId, data._1)
    )
    // 找出用户已经打过分的电影
    val movieRated = dataForRating.map{
      line => val fields = line.split(",")
        (fields(0).toInt, fields(1).toInt)
    }.filter(_._1 == userId)

    // 选出前十名
    val predictionForUser_1 = model.predict(userProductFor_1).map{
      case Rating(user, product, rate) => (product, (user, rate))
    }.join(movieRDD).sortBy(-_._2._1._2).take(10)

    predictionForUser_1.foreach(
      data =>
        println("User: " + data._2._1._1 + " MovieId: " + data._1 + " MovieName: " + data._2._2._1 + " MovieType: " + data._2._2._2 + " Rate: " + data._2._1._2 )
    )

    println()
    println("--------------数据比对--------------")

    val userProductFor_test = movieRated.map(
      // user, product
      data => (userId, data._2)
    )
    val predictionForUser_test = model.predict(userProductFor_test).map{
      case Rating(user, product, rate) => (user, product, rate)
    }.foreach{
      data => println("User: " + data._1 + " MovieId: " + data._2 + " Rate: " + data._3)
    }
  }

}
