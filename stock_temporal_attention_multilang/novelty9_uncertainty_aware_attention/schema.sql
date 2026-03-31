-- MySQL schema for stock temporal attention
CREATE DATABASE IF NOT EXISTS stock_attention;
USE stock_attention;

CREATE TABLE IF NOT EXISTS stock_series (
  id INT AUTO_INCREMENT PRIMARY KEY,
  ts INT,
  price DOUBLE,
  volume DOUBLE,
  feature1 DOUBLE,
  feature2 DOUBLE
);

CREATE TABLE IF NOT EXISTS attention_weights (
  id INT AUTO_INCREMENT PRIMARY KEY,
  novelty VARCHAR(100),
  ts INT,
  weight DOUBLE
);
