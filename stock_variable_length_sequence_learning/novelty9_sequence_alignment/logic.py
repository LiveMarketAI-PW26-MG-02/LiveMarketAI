def main_logic(data):
    # simple normalization for alignment
    m = sum(data)/len(data)
    return [(d - m) for d in data]
