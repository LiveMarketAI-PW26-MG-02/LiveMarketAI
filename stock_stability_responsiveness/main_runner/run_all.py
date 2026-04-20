import os
folders=['novelty1_adaptive_smoothing', 'novelty2_dual_model', 'novelty3_responsiveness_scaling', 'novelty4_change_detection', 'novelty5_temporal_inertia', 'novelty6_confidence_weighting', 'novelty7_multi_window_consensus', 'novelty8_dynamic_threshold', 'novelty9_feedback_tuning', 'novelty10_comparative_evaluation']
for f in folders:
 os.system(f'python {f}/run.py')