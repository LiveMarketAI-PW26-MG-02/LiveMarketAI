from utils import time_gaps
def main_logic(prices, times):
    gaps = time_gaps(times)
    anomalies = [g for g in gaps if g > 2]
    return anomalies
