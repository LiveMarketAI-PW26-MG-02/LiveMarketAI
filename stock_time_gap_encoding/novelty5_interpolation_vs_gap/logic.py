def main_logic(prices, times):
    interp = sum(prices)/len(prices)
    preserve = prices[-1]
    return {"interpolated": interp, "gap_preserved": preserve}
