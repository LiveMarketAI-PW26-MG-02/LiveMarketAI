from utils import time_gaps
def main_logic(prices, times):
    gaps = time_gaps(times)
    hidden = 0
    for p,g in zip(prices,gaps):
        hidden = 0.5*hidden + (1/(1+g))*p
    return hidden
