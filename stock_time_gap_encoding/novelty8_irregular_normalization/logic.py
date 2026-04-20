from utils import normalize, time_gaps
def main_logic(prices, times):
    gaps = time_gaps(times)
    norm_prices = normalize(prices)
    return list(zip(norm_prices, gaps))
