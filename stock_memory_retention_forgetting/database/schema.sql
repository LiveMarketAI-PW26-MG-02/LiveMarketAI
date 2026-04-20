
CREATE DATABASE IF NOT EXISTS stock_memory;
USE stock_memory;

CREATE TABLE memory_logs(
 id INT AUTO_INCREMENT PRIMARY KEY,
 novelty VARCHAR(50),
 value FLOAT
);
