import numpy as np

def compute_attention(data):
    # placeholder temporal attention
    weights = np.linspace(0.1, 1.0, len(data))
    weights = weights / weights.sum()
    return weights.tolist()

if __name__ == "__main__":
    data = [100,102,101,105,110]
    print("Python attention:", compute_attention(data))
