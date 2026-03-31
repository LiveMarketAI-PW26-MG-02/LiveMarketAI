from utils import importance_scores

def main_logic(data):
    scores = importance_scores(data)
    weighted = sum(d*s for d,s in zip(data, scores)) / (sum(scores)+1e-6)
    return weighted
