# RD-GNN — 08 Risk Scoring

## Module: Risk Diffusion Graph Network

### Overview
Ensemble risk scoring, regulatory alignment, and alert generation

This subfolder implements **RD-GNN** capabilities for:
> *Regulatory network graph for compliance-aware risk assessment*

---

## Novelty Bullets (Module Level)

1. Systemic risk propagation modeling via diffusion GNN on financial networks
2. Contagion channel identification through edge betweenness centrality analysis
3. Too-big-to-fail node detection using eigenvector centrality in risk graphs
4. Cascade failure simulation through iterative graph risk propagation
5. Risk absorption capacity modeling via graph community structure analysis
6. Cross-asset contagion graph with correlation-weighted risk edges
7. Liquidity risk diffusion through interbank network graph modeling
8. Regulatory network graph for compliance-aware risk assessment
9. Stress testing via targeted node removal and graph resilience metrics
10. Early warning system using graph-based leading risk indicator propagation

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
cd RD-GNN/08_risk_scoring
python main.py
```

Or via the module demo runner:
```bash
cd RD-GNN/10_demo_runner
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
      Generated 4 alerts → outputs/rd_gnn_demo/alerts.json
```

### risk_scores.csv (sample)
```
asset_id,risk_score,severity,label,module
ASSET_000,0.8731,critical,1,RD-GNN
ASSET_001,0.7245,high,1,RD-GNN
ASSET_002,0.2134,low,0,RD-GNN
```

### alerts.json (sample)
```json
{
  "module": "RD-GNN",
  "total": 4,
  "alerts": [
    {
      "alert_id": "ALERT_RD-GNN_00001",
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
   RD-GNN GNN Forward Pass
         ↓
   Risk Score Computation
         ↓
   Alert Generation & XAI
         ↓
   Output: risk_scores.csv + alerts.json + explanation.txt
```

---

## Research Context

**Risk Diffusion Graph Network** addresses the key challenge of explainable, graph-based
financial market manipulation detection. This subfolder (08_risk_scoring)
contributes to the research by implementing:

*Regulatory network graph for compliance-aware risk assessment*

The implementation maps directly to the novelty bullet above, providing
a concrete computational realization of the theoretical contribution.

---

*Part of the 17th_may_2026 Research Project | RD-GNN Module*
