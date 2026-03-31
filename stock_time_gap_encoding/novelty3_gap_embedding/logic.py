from utils import time_gaps
def main_logic(prices, times):
    gaps = time_gaps(times)
    return [ (p, g, g*0.5) for p,g in zip(prices,gaps) ]
