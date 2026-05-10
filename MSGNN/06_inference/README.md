# MSGNN — 06 Inference

## Module: Multi-Scale Graph Neural Network

### Overview
Real-time inference pipeline with streaming graph updates

This subfolder implements **MSGNN** capabilities for:
> *Scale transition anomaly detection for manipulation phase identification*

---

## Novelty Bullets (Module Level)

1. Hierarchical graph pooling capturing market structure at multiple scales
2. Cross-scale attention mechanism bridging tick and daily patterns
3. Scale-specific feature extraction with resolution-aware GNN layers
4. Fractal pattern detection via self-similar graph substructure analysis
5. Multi-resolution wavelet features integrated into graph node embeddings
6. Scale transition anomaly detection for manipulation phase identification
7. Upsampling and downsampling graph operations for scale coherence
8. Scale-invariant manipulation signature learning across timeframes
9. Multi-scale risk aggregation with scale-weighted ensemble fusion
10. Explanation generation identifying scale at which manipulation is strongest

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
cd MSGNN/06_inference
python main.py
```

Or via the module demo runner:
```bash
cd MSGNN/10_demo_runner
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
      Generated 4 alerts → outputs/msgnn_demo/alerts.json
```

### risk_scores.csv (sample)
```
asset_id,risk_score,severity,label,module
ASSET_000,0.8731,critical,1,MSGNN
ASSET_001,0.7245,high,1,MSGNN
ASSET_002,0.2134,low,0,MSGNN
```

### alerts.json (sample)
```json
{
  "module": "MSGNN",
  "total": 4,
  "alerts": [
    {
      "alert_id": "ALERT_MSGNN_00001",
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
   MSGNN GNN Forward Pass
         ↓
   Risk Score Computation
         ↓
   Alert Generation & XAI
         ↓
   Output: risk_scores.csv + alerts.json + explanation.txt
```

---

## Research Context

**Multi-Scale Graph Neural Network** addresses the key challenge of explainable, graph-based
financial market manipulation detection. This subfolder (06_inference)
contributes to the research by implementing:

*Scale transition anomaly detection for manipulation phase identification*

The implementation maps directly to the novelty bullet above, providing
a concrete computational realization of the theoretical contribution.

---

*Part of the 17th_may_2026 Research Project | MSGNN Module*
