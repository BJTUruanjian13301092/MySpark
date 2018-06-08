package com.example.sparktest.scala

object HelloScala {
  def main(args: Array[String]): Unit = {
    val str:String = printMyString
    println(str)
    println("Scala : Hello world!")
  }

  def printMyString: String = {
    return "This is my printed String"
  }
}
