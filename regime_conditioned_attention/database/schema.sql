CREATE DATABASE IF NOT EXISTS regime_attention;
USE regime_attention;
CREATE TABLE experiments (
    id BIGINT AUTO_INCREMENT PRIMARY KEY, run_name VARCHAR(128),
    d_model INT DEFAULT 256, n_regimes INT DEFAULT 4, n_sources INT DEFAULT 8,
    lr FLOAT DEFAULT 3e-4, epochs INT DEFAULT 50, seed INT DEFAULT 42,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, INDEX idx_run(run_name));
CREATE TABLE epoch_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY, experiment_id BIGINT,
    epoch INT, train_loss FLOAT, val_loss FLOAT, entropy_reg FLOAT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    INDEX idx_exp_epoch(experiment_id,epoch));
CREATE TABLE attention_snapshots (
    id BIGINT AUTO_INCREMENT PRIMARY KEY, experiment_id BIGINT,
    epoch INT, regime_id INT, source_idx INT, alpha_weight FLOAT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id));
