USE regime_attention;
SELECT e.run_name, MIN(m.val_loss) AS best_val, COUNT(m.epoch) AS epochs_run
FROM experiments e JOIN epoch_metrics m ON m.experiment_id=e.id
GROUP BY e.id,e.run_name ORDER BY best_val;
SELECT a.regime_id, a.source_idx, AVG(a.alpha_weight) AS mean_alpha, STDDEV(a.alpha_weight) AS std_alpha
FROM attention_snapshots a GROUP BY a.regime_id,a.source_idx ORDER BY a.regime_id,a.source_idx;
SELECT regime_id, source_idx, AVG(alpha_weight) AS w FROM attention_snapshots GROUP BY regime_id,source_idx ORDER BY regime_id,w DESC LIMIT 12;
