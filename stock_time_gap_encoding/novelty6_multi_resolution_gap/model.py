import numpy as np

class StockModel:
    def predict(self, seq):
        return np.mean(seq)
