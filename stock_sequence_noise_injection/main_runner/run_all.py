import os

novelties = ['novelty1_gaussian_noise', 'novelty2_variance_scaled_noise', 'novelty3_snr_metric', 'novelty4_temporal_decay_noise', 'novelty5_correlated_noise', 'novelty6_sensitivity_curve', 'novelty7_quantile_robustness', 'novelty8_bias_variance_decomp', 'novelty9_monte_carlo', 'novelty10_stat_tests']

for n in novelties:
    print("\nRunning", n)
    os.system(f"python {n}/logic.py")
    if os.path.exists(f"{n}/logic.exe"):
        os.system(f"{n}/logic.exe")
    os.system(f"java -cp {n} Logic 2>nul")
