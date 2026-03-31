import os
folders=['novelty1_gradient_clipping', 'novelty2_variance_scaling', 'novelty3_time_decay', 'novelty4_jacobian_conditioning', 'novelty5_noise_injection', 'novelty6_second_order_correction', 'novelty7_layerwise_normalization', 'novelty8_temporal_rescaling', 'novelty9_dynamic_lr', 'novelty10_comparative_evaluation']
for f in folders:
 os.system(f'python {f}/logic.py')