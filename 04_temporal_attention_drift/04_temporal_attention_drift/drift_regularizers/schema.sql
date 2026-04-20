-- Schema for temporal_drift.drift_regularizers
CREATE DATABASE IF NOT EXISTS temporal_drift;
USE temporal_drift;

CREATE TABLE IF NOT EXISTS drift_regularizers (
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
