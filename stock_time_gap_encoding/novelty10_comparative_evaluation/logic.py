def main_logic(prices, times):
    uniform = sum(prices)/len(prices)
    gap_aware = prices[-1]
    return {"uniform": uniform, "gap_aware": gap_aware}
