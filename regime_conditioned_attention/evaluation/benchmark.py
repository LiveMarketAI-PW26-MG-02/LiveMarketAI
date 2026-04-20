"""Benchmark attention across configs."""
import torch, time
from attention_equation import RegimeConditionedAttention

CONFIGS=[dict(d_model=128,n_regimes=4,n_sources=8),dict(d_model=256,n_regimes=4,n_sources=8),
         dict(d_model=256,n_regimes=8,n_sources=8),dict(d_model=256,n_regimes=4,n_sources=16),
         dict(d_model=512,n_regimes=4,n_sources=8)]

def run_benchmark():
    results=[]
    for cfg in CONFIGS:
        model=RegimeConditionedAttention(**cfg).eval()
        X=torch.randn(64,cfg["n_sources"],cfg["d_model"]); R=torch.randint(0,cfg["n_regimes"],(64,))
        for _ in range(10):
            with torch.no_grad(): model(X,R)
        t0=time.perf_counter()
        for _ in range(100):
            with torch.no_grad(): model(X,R)
        results.append({**cfg,"latency_ms":round((time.perf_counter()-t0)/100*1000,3)})
    return results

if __name__=="__main__":
    for r in run_benchmark(): print(r)
