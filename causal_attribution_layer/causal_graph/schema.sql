-- Schema for causal_attribution.causal_graph
CREATE DATABASE IF NOT EXISTS causal_attribution;
USE causal_attribution;

CREATE TABLE IF NOT EXISTS causal_graph (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_id BIGINT,
    run_name      VARCHAR(128),
    value         FLOAT,
    metric_name   VARCHAR(64),
    epoch         INT,
    regime_id     INT DEFAULT NULL,
    
    notes         TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_run (run_name),
    INDEX idx_epoch (experiment_id, epoch)
);
