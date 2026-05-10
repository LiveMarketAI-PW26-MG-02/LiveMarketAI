# GRL-Fusion — 08 Risk Scoring

## Module: Graph Reinforcement Learning Fusion

### Overview
Ensemble risk scoring, regulatory alignment, and alert generation

This subfolder implements **GRL-Fusion** capabilities for:
> *Meta-learning graph RL for rapid adaptation to new manipulation schemes*

---

## Novelty Bullets (Module Level)

1. RL policy network operating on GNN-generated graph state representations
2. Reward function design incorporating precision, recall, and regulatory cost
3. Multi-agent graph RL for competitive and cooperative manipulation detection
4. Graph-structured action space for adaptive module weight selection
5. Experience replay with graph-aware prioritized sampling
6. Policy gradient optimization with graph topology regularization
7. Curriculum learning strategy for progressive manipulation complexity
8. Meta-learning graph RL for rapid adaptation to new manipulation schemes
9. Adversarial RL training with simulated manipulator agents
10. Explainable RL policy via graph attention visualization of fusion decisions

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
cd GRL-Fusion/08_risk_scoring
python main.py
```

Or via the module demo runner:
```bash
cd GRL-Fusion/10_demo_runner
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
      Generated 4 alerts → outputs/grl_fusion_demo/alerts.json
```

### risk_scores.csv (sample)
```
asset_id,risk_score,severity,label,module
ASSET_000,0.8731,critical,1,GRL-Fusion
ASSET_001,0.7245,high,1,GRL-Fusion
ASSET_002,0.2134,low,0,GRL-Fusion
```

### alerts.json (sample)
```json
{
  "module": "GRL-Fusion",
  "total": 4,
  "alerts": [
    {
      "alert_id": "ALERT_GRL-Fusion_00001",
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
   GRL-Fusion GNN Forward Pass
         ↓
   Risk Score Computation
         ↓
   Alert Generation & XAI
         ↓
   Output: risk_scores.csv + alerts.json + explanation.txt
```

---

## Research Context

**Graph Reinforcement Learning Fusion** addresses the key challenge of explainable, graph-based
financial market manipulation detection. This subfolder (08_risk_scoring)
contributes to the research by implementing:

*Meta-learning graph RL for rapid adaptation to new manipulation schemes*

The implementation maps directly to the novelty bullet above, providing
a concrete computational realization of the theoretical contribution.

---

*Part of the 17th_may_2026 Research Project | GRL-Fusion Module*
