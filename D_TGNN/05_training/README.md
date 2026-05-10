# D-TGNN — 05 Training

## Module: Dynamic Temporal Graph Neural Network

### Overview
Training loop, loss functions, optimization, and hyperparameter management

This subfolder implements **D-TGNN** capabilities for:
> *Multi-granularity temporal aggregation across tick/minute/daily*

---

## Novelty Bullets (Module Level)

1. Time-varying graph topology with sliding window edge creation
2. Temporal node embeddings using GRU memory cells
3. Causal temporal attention preventing future data leakage
4. Event-driven graph restructuring on significant price moves
5. Multi-granularity temporal aggregation across tick/minute/daily
6. Anomaly-aware temporal edge weighting for suspicious activity
7. Temporal pattern library for known manipulation sequences
8. Graph snapshot differencing to detect topology manipulation
9. Forecasting-augmented GNN predicting next manipulation phase
10. Time-series explainability with temporal saliency maps

---

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — full pipeline execution |
| `model.py` / `engine.py` | Core logic, classes, and algorithms |
| `neo4j_schema.cypher` | Neo4j schema: constraints, indexes, node/rel creation |
| `neo4j_queries.cypher` | Operational Cypher queries for analytics |
| `config.json` | Configuration parameters |
| `utils.py` | Shared utility functions |
| `README.md` | This file |

---

## How to Run

```bash
# From project root (with venv activated):
cd D-TGNN/05_training
python main.py
```

Or via the module demo runner:
```bash
cd D-TGNN/10_demo_runner
python main.py
```

---

## Neo4j Graph Structure

### Nodes Used
- **Asset** — Financial instrument with risk score
- **Influencer** — Social/news source affecting sentiment
- **MarketRegime** — Detected market condition window
- **OrderBookLevel** — Limit order book price level
- **Signal** — Detected anomaly signal
- **RiskAlert** — Actionable manipulation alert
- **Explanation** — XAI explanation for model decision

### Relationships Used
- `INFLUENCES` — Influencer → Asset sentiment impact
- `CORRELATES_WITH` — Asset ↔ Asset price co-movement
- `HAS_REGIME` — Asset → MarketRegime membership
- `PROPAGATES_RISK` — Risk contagion between assets/signals
- `TRIGGERS_ALERT` — Signal → RiskAlert activation
- `CAUSES_SIGNAL` — Asset/Influencer/Regime → Signal

---

## Sample Output

```
[1/5] Generating synthetic market data...
      → 98,800 rows × 18 columns
[2/5] Building graph...
      Correlation graph: 20 nodes, 47 edges
[3/5] GNN inference...
      Flagged: 4 / 20 assets (20.0%)
[4/5] Risk scoring...
      Mean risk: 0.3842 | Critical: 1 | High: 3
[5/5] Generating alerts...
      Generated 4 alerts → outputs/d_tgnn_demo/alerts.json
```

### risk_scores.csv (sample)
```
asset_id,risk_score,severity,label,module
ASSET_000,0.8731,critical,1,D-TGNN
ASSET_001,0.7245,high,1,D-TGNN
ASSET_002,0.2134,low,0,D-TGNN
```

### alerts.json (sample)
```json
{
  "module": "D-TGNN",
  "total": 4,
  "alerts": [
    {
      "alert_id": "ALERT_D-TGNN_00001",
      "asset_id": "ASSET_000",
      "risk_score": 0.8731,
      "severity": "critical",
      "manipulation_type": "Pump-and-Dump"
    }
  ]
}
```

---

## Architecture

```
Raw Data (OHLCV + Sentiment + OrderBook)
         ↓
   Feature Engineering
         ↓
   Graph Construction (NetworkX / Neo4j)
         ↓
   D-TGNN GNN Forward Pass
         ↓
   Risk Score Computation
         ↓
   Alert Generation & XAI
         ↓
   Output: risk_scores.csv + alerts.json + explanation.txt
```

---

## Research Context

**Dynamic Temporal Graph Neural Network** addresses the key challenge of explainable, graph-based
financial market manipulation detection. This subfolder (05_training)
contributes to the research by implementing:

*Multi-granularity temporal aggregation across tick/minute/daily*

The implementation maps directly to the novelty bullet above, providing
a concrete computational realization of the theoretical contribution.

---

*Part of the 17th_may_2026 Research Project | D-TGNN Module*
