import os

folders = ['n1_importance_compression', 'n2_autoencoder_compression', 'n3_temporal_saliency', 'n4_multi_resolution', 'n5_dynamic_pruning', 'n6_redundancy_detection', 'n7_context_aware', 'n8_compression_aware_training', 'n9_feature_preserving', 'n10_comparative_evaluation']

for f in folders:
    print("Running", f)
    os.system(f"python {f}/module.py")
