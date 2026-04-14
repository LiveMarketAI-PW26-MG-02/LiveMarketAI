-- Schema for nsras_clf.classifier_head
CREATE DATABASE IF NOT EXISTS nsras_clf;
USE nsras_clf;

CREATE TABLE IF NOT EXISTS classifier_head (
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
