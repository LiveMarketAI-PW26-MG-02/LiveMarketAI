from utils import time_gaps
import numpy as np
def main_logic(prices, times):
    gaps = time_gaps(times)
    weights = np.exp(-0.1 * gaps)
    return list(weights * prices)
