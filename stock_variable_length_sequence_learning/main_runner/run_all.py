import os
folders=['novelty1_dynamic_sequence_length', 'novelty2_attention_guided_selection', 'novelty3_event_driven_expansion', 'novelty4_padding_free_pipeline', 'novelty5_hierarchical_modeling', 'novelty6_sequence_importance_scoring', 'novelty7_adaptive_truncation', 'novelty8_multi_resolution_sequences', 'novelty9_sequence_alignment', 'novelty10_comparative_evaluation']
for f in folders:
 print('Running', f)
 os.system(f'python {f}/run.py')