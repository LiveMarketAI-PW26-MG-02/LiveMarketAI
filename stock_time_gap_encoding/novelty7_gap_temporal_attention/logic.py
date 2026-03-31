from utils import time_gaps
import numpy as np
def main_logic(prices, times):
    gaps = time_gaps(times)
    attn = np.exp(-gaps)
    return list(attn * prices)
