import numpy as np

class StockSequenceModel:
    def predict(self, seq):
        return float(np.mean(seq))
