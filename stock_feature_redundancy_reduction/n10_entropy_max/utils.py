import numpy as np
def cov(X): return np.cov(X, rowvar=False)
def entropy(x):
    p = np.histogram(x, bins=10, density=True)[0] + 1e-9
    return -np.sum(p*np.log(p))
