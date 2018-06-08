package com.example.sparktest.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class BaseController {

    public Logger logger = LoggerFactory.getLogger(this.getClass());
}
