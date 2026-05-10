# UGNN — 02 Graph Schema

## Module: Uncertainty-Aware Graph Neural Network

### Overview
Neo4j graph schema definition, node/relationship types, and population scripts

This subfolder implements **UGNN** capabilities for:
> *Epistemic uncertainty quantification via Bayesian graph neural networks*

---

## Novelty Bullets (Module Level)

1. Monte Carlo dropout uncertainty estimation in GNN message passing
2. Epistemic uncertainty quantification via Bayesian graph neural networks
3. Aleatoric uncertainty modeling from noisy financial label distributions
4. Out-of-distribution detection using graph topology distance metrics
5. Uncertainty-calibrated risk thresholds preventing false alarm inflation
6. Conformal prediction intervals for GNN manipulation probability bounds
7. Graph-level uncertainty propagation from node to global risk scores
8. Active learning graph sampling guided by uncertainty for data efficiency
9. Uncertainty-aware explanation generation with confidence intervals
10. Risk communication framework translating uncertainty to regulatory language

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
cd UGNN/02_graph_schema
python main.py
```

Or via the module demo runner:
```bash
cd UGNN/10_demo_runner
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
      Generated 4 alerts → outputs/ugnn_demo/alerts.json
```

### risk_scores.csv (sample)
```
asset_id,risk_score,severity,label,module
ASSET_000,0.8731,critical,1,UGNN
ASSET_001,0.7245,high,1,UGNN
ASSET_002,0.2134,low,0,UGNN
```

### alerts.json (sample)
```json
{
  "module": "UGNN",
  "total": 4,
  "alerts": [
    {
      "alert_id": "ALERT_UGNN_00001",
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
   UGNN GNN Forward Pass
         ↓
   Risk Score Computation
         ↓
   Alert Generation & XAI
         ↓
   Output: risk_scores.csv + alerts.json + explanation.txt
```

---

## Research Context

**Uncertainty-Aware Graph Neural Network** addresses the key challenge of explainable, graph-based
financial market manipulation detection. This subfolder (02_graph_schema)
contributes to the research by implementing:

*Epistemic uncertainty quantification via Bayesian graph neural networks*

The implementation maps directly to the novelty bullet above, providing
a concrete computational realization of the theoretical contribution.

---

*Part of the 17th_may_2026 Research Project | UGNN Module*
