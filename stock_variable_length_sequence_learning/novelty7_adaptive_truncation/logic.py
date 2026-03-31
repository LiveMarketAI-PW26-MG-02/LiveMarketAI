from utils import importance_scores

def main_logic(data):
    scores = importance_scores(data)
    threshold = sum(scores)/len(scores)
    return [d for d,s in zip(data, scores) if s >= threshold]
