# EP-GNN — 09 Evaluation

## Module: Emotion Propagation Graph Neural Network

### Overview
Comprehensive evaluation: metrics, regime-stratified analysis, calibration

This subfolder implements **EP-GNN** capabilities for:
> *Rumor propagation path reconstruction using backward graph traversal*

---

## Novelty Bullets (Module Level)

1. Sentiment contagion modeling via SIR-inspired graph propagation
2. Influencer network graph with follower-weighted message passing
3. Bot account detection through network topology anomaly analysis
4. Coordinated inauthentic behavior detection via posting time correlations
5. Cross-platform sentiment fusion from multiple social graph sources
6. Emotion velocity tracking for sudden sentiment surge detection
7. Lexical manipulation fingerprinting via shared phrase graph clustering
8. Opinion leader identification through centrality-weighted GNN layers
9. Rumor propagation path reconstruction using backward graph traversal
10. Sentiment-price decoupling detection for manipulation alert generation

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
cd EP-GNN/09_evaluation
python main.py
```

Or via the module demo runner:
```bash
cd EP-GNN/10_demo_runner
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
      Generated 4 alerts → outputs/ep_gnn_demo/alerts.json
```

### risk_scores.csv (sample)
```
asset_id,risk_score,severity,label,module
ASSET_000,0.8731,critical,1,EP-GNN
ASSET_001,0.7245,high,1,EP-GNN
ASSET_002,0.2134,low,0,EP-GNN
```

### alerts.json (sample)
```json
{
  "module": "EP-GNN",
  "total": 4,
  "alerts": [
    {
      "alert_id": "ALERT_EP-GNN_00001",
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
   EP-GNN GNN Forward Pass
         ↓
   Risk Score Computation
         ↓
   Alert Generation & XAI
         ↓
   Output: risk_scores.csv + alerts.json + explanation.txt
```

---

## Research Context

**Emotion Propagation Graph Neural Network** addresses the key challenge of explainable, graph-based
financial market manipulation detection. This subfolder (09_evaluation)
contributes to the research by implementing:

*Rumor propagation path reconstruction using backward graph traversal*

The implementation maps directly to the novelty bullet above, providing
a concrete computational realization of the theoretical contribution.

---

*Part of the 17th_may_2026 Research Project | EP-GNN Module*
