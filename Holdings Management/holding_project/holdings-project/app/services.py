def get_price(instrument_id):
    prices = {
        "AAPL": 180,
        "TSLA": 240,
        "GOOG": 150
    }
    return prices.get(instrument_id, 100)