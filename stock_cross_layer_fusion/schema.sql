CREATE DATABASE IF NOT EXISTS stock_fusion;
USE stock_fusion;

CREATE TABLE fusion_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    novelty VARCHAR(100),
    value DOUBLE
);
