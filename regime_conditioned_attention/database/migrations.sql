USE regime_attention;
ALTER TABLE epoch_metrics ADD COLUMN regime_acc FLOAT DEFAULT NULL;
CREATE TABLE hyperparameter_search (
    id BIGINT AUTO_INCREMENT PRIMARY KEY, experiment_id BIGINT,
    param_name VARCHAR(64), param_value VARCHAR(128), val_loss FLOAT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id));
