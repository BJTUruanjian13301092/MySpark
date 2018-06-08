package com.example.sparktest;

import com.example.sparktest.service.SparkService;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringBootTest(classes = SparktestApplication.class)
public class SparkTest {

    @Autowired
    SparkService sparkService;

    @Test
    public void testSparkService(){
        sparkService.getSparkSqlList();
    }
}
