import os

novelties = ['n1_rank_maximization', 'n2_gram_schmidt', 'n3_mutual_information', 'n4_eigen_filtering', 'n5_pca_variance', 'n6_cauchy_schwarz', 'n7_sparse_l1', 'n8_matrix_factorization', 'n9_subspace_intersection', 'n10_entropy_max']

for n in novelties:
    print("Running", n)
    os.system(f"python {n}/logic.py")
