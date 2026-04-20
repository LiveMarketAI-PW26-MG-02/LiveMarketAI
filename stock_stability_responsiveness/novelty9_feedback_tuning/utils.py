import numpy as np
def calculate_volatility(data): return np.std(data)
def confidence_score(data): return 1/(1+np.std(data))
