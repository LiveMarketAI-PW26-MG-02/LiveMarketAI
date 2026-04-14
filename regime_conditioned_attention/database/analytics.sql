USE regime_attention;
SELECT a1.regime_id AS ra, a2.regime_id AS rb,
       SUM(ABS(AVG(a1.alpha_weight)-AVG(a2.alpha_weight))) AS l1_div
FROM attention_snapshots a1
JOIN attention_snapshots a2 ON a1.source_idx=a2.source_idx AND a1.experiment_id=a2.experiment_id AND a1.regime_id<a2.regime_id
GROUP BY a1.regime_id,a2.regime_id ORDER BY l1_div DESC;
SELECT e.run_name, m.epoch AS conv_epoch, m.val_loss FROM epoch_metrics m JOIN experiments e ON e.id=m.experiment_id
WHERE m.val_loss=(SELECT MIN(m2.val_loss) FROM epoch_metrics m2 WHERE m2.experiment_id=m.experiment_id);
