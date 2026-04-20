import numpy as np

def volatility(x):
    return float(np.std(x))

def event_magnitude(x):
    if len(x) < 2: return 0.0
    return float(abs(x[-1] - x[-2]))

def importance_scores(x):
    # simple proxy: distance from mean
    m = np.mean(x)
    return [float(abs(v - m)) for v in x]
