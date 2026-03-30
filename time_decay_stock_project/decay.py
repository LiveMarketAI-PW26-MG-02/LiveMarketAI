
import numpy as np

def exponential_decay(data, alpha=0.5):
    return [np.exp(-alpha*i) for i in range(len(data))]

def adaptive_decay(volatility):
    return 0.1 if volatility < 0.5 else 0.5
