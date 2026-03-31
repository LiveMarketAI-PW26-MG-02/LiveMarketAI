import os
folders=['novelty1_time_gap_encoding', 'novelty2_decay_weighting', 'novelty3_gap_embedding', 'novelty4_gap_aware_rnn', 'novelty5_interpolation_vs_gap', 'novelty6_multi_resolution_gap', 'novelty7_gap_temporal_attention', 'novelty8_irregular_normalization', 'novelty9_gap_anomaly_detection', 'novelty10_comparative_evaluation']
for f in folders:
 print('Running',f); os.system(f'python {f}/run.py')