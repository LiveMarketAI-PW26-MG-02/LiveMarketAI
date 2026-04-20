import os
folders = ['novelty1_temporal_attention_weighting', 'novelty2_volatility_aware_attention', 'novelty3_recency_biased_attention', 'novelty4_multi_head_attention', 'novelty5_context_conditioned_attention', 'novelty6_learnable_decay_attention', 'novelty7_cross_feature_attention', 'novelty8_attention_consistency_regularization', 'novelty9_uncertainty_aware_attention', 'novelty10_comparative_evaluation']
for f in folders:
    print('Running', f)
    os.system(f'python {f}/attention.py')
