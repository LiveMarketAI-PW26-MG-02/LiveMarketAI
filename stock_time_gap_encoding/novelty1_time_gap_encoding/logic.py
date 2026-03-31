from utils import time_gaps
def main_logic(prices, times):
    gaps = time_gaps(times)
    return list(zip(prices, gaps))
