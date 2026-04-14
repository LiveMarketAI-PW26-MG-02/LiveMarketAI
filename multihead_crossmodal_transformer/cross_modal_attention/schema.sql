-- Schema for crossmodal_transformer.cross_modal_attention
CREATE DATABASE IF NOT EXISTS crossmodal_transformer;
USE crossmodal_transformer;

CREATE TABLE IF NOT EXISTS cross_modal_attention (
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
