
from decay import exponential_decay, adaptive_decay

def run():
    print("Running Time-Decayed Stock Weighting System")
    data = [1,2,3,4,5]
    weights = exponential_decay(data)
    print("Weights:", weights)

if __name__ == "__main__":
    run()
