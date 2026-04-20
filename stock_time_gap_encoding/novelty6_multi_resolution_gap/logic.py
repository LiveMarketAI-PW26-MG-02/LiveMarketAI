from utils import time_gaps
def main_logic(prices, times):
    gaps = time_gaps(times)
    short = sum(gaps[:len(gaps)//2])
    long = sum(gaps[len(gaps)//2:])
    return {"short_gap": short, "long_gap": long}
