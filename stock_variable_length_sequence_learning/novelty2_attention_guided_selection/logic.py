from utils import importance_scores

def main_logic(data):
    scores = importance_scores(data)
    idx = sorted(range(len(data)), key=lambda i: scores[i], reverse=True)[:max(1, len(data)//2)]
    idx.sort()
    return [data[i] for i in idx]
