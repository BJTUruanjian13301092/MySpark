package com.example.sparktest.service;

import com.example.sparktest.scala.spark.MySpark;
import com.example.sparktest.scala.spark.Words;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SparkService extends BaseService {

    public void getSparkSqlList(){
        List<Words> listWords = (List<Words>)MySpark.sparkSQLTest();
        for(Words words : listWords){
            System.out.println("words: " + words.word() + "  counts: " + words.count());
        }
    }
}
