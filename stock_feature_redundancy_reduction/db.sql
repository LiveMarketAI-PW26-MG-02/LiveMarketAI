CREATE DATABASE IF NOT EXISTS stock_features;
USE stock_features;

CREATE TABLE IF NOT EXISTS feature_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    novelty VARCHAR(50),
    metric DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
