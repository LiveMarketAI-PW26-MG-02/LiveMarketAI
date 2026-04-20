
from modules.persistence import enforce_min_duration
from modules.penalty import transition_penalty
from modules.smoothing import temporal_smoothing
from modules.hysteresis import hysteresis_classification
from modules.confidence import confidence_override
from modules.regime import regime_check
from modules.memory import time_decay_memory
from modules.consensus import multi_window_consensus
from modules.probabilistic import probabilistic_transition
from modules.evaluation import evaluate

def run():
    print("Running Stock State Persistence Modeling...")
    evaluate()

if __name__ == "__main__":
    run()
