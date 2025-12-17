package com.advdb.front;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class FrontApplication {

	private static final Logger logger = LoggerFactory.getLogger(FrontApplication.class);
	public static void main(String[] args) {
		SpringApplication.run(FrontApplication.class, args);
		logger.info("Je tourne sur http://localhost:7777");
	}

}