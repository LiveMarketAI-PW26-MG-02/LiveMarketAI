USE model_comparison;

-- Aggregate metrics per run
SELECT run_name,
       AVG(value)    AS mean_value,
       MIN(value)    AS min_value,
       MAX(value)    AS max_value,
       STDDEV(value) AS std_value,
       COUNT(*)      AS n_records
FROM lstm_models
WHERE metric_name = 'val_loss'
GROUP BY run_name
ORDER BY mean_value;

-- Best epoch per experiment
SELECT experiment_id, epoch, MIN(value) AS best_val
FROM lstm_models
WHERE metric_name = 'val_loss'
GROUP BY experiment_id, epoch
ORDER BY best_val
LIMIT 20;

-- Per-regime breakdown
SELECT regime_id, metric_name, AVG(value) AS avg_val, STDDEV(value) AS std_val
FROM lstm_models
GROUP BY regime_id, metric_name
ORDER BY regime_id, metric_name;

-- Recent runs
SELECT * FROM lstm_models ORDER BY created_at DESC LIMIT 50;
