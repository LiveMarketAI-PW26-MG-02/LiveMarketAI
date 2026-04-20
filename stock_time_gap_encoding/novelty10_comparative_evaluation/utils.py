import numpy as np

def time_gaps(times):
    return np.diff(times, prepend=times[0])

def normalize(x):
    x = np.array(x)
    return (x - x.mean())/(x.std()+1e-6)
