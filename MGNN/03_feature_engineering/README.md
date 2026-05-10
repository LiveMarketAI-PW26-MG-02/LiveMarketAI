# MGNN — 03 Feature Engineering

## Module: Microstructure Graph Neural Network

### Overview
Feature extraction, transformation, and graph-aware feature computation

This subfolder implements **MGNN** capabilities for:
> *Spoofing pattern detection using order cancellation graph substructure*

---

## Novelty Bullets (Module Level)

1. Order book level graph with bid-ask hierarchy as tree structure
2. Trade-to-quote ratio anomaly detection via graph density analysis
3. Spoofing pattern detection using order cancellation graph substructure
4. Layering detection via multi-level order placement graph motifs
5. Quote stuffing identification through high-frequency edge creation bursts
6. Momentum ignition detection via directional order flow graph analysis
7. Market depth imbalance encoding as asymmetric graph weights
8. Cross-exchange order book synchronization graph for manipulation arbitrage
9. Iceberg order detection via hidden liquidity graph inference
10. Microstructure alpha decay modeling through temporal order graph evolution

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
cd MGNN/03_feature_engineering
python main.py
```

Or via the module demo runner:
```bash
cd MGNN/10_demo_runner
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
      Generated 4 alerts → outputs/mgnn_demo/alerts.json
```

### risk_scores.csv (sample)
```
asset_id,risk_score,severity,label,module
ASSET_000,0.8731,critical,1,MGNN
ASSET_001,0.7245,high,1,MGNN
ASSET_002,0.2134,low,0,MGNN
```

### alerts.json (sample)
```json
{
  "module": "MGNN",
  "total": 4,
  "alerts": [
    {
      "alert_id": "ALERT_MGNN_00001",
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
   MGNN GNN Forward Pass
         ↓
   Risk Score Computation
         ↓
   Alert Generation & XAI
         ↓
   Output: risk_scores.csv + alerts.json + explanation.txt
```

---

## Research Context

**Microstructure Graph Neural Network** addresses the key challenge of explainable, graph-based
financial market manipulation detection. This subfolder (03_feature_engineering)
contributes to the research by implementing:

*Spoofing pattern detection using order cancellation graph substructure*

The implementation maps directly to the novelty bullet above, providing
a concrete computational realization of the theoretical contribution.

---

*Part of the 17th_may_2026 Research Project | MGNN Module*
