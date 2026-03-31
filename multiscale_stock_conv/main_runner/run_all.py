import os

novelties = ['novelty1_scale_invariant', 'novelty2_extremal_scale', 'novelty3_wavelet_decomposition', 'novelty4_minkowski_bound', 'novelty5_spectral_partition', 'novelty6_recursive_multiscale', 'novelty7_energy_conservation', 'novelty8_convex_optimization', 'novelty9_combinatorial_interactions', 'novelty10_spectral_radius_stability']

for n in novelties:
    print(f"Running {n}...")
    os.system(f"python {n}/logic.py")
