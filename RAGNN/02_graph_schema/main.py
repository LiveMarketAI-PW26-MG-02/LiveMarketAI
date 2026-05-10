#!/usr/bin/env python3
"""
Regime-Aware Graph Neural Network (RAGNN) - Graph Schema Module
Defines, validates, and loads Neo4j graph schema for manipulation detection.
Implements all required node types, relationship types, and constraint creation.
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs/ragnn_graph"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# NODE TYPE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

NODE_TYPES = {
    "Asset": {
        "description": "A financial instrument (stock, ETF, crypto, etc.)",
        "required_props": ["assetId", "symbol", "sector", "exchange"],
        "optional_props": ["marketCap", "listingDate", "currency", "manipulationRiskScore"],
        "index_props": ["assetId", "symbol"],
        "unique_constraint": "assetId",
    },
    "Influencer": {
        "description": "Social media account, news source, or market commentator",
        "required_props": ["influencerId", "platform", "followerCount", "influencerType"],
        "optional_props": ["avgSentiment", "postCount", "engagementRate", "isSuspicious"],
        "index_props": ["influencerId"],
        "unique_constraint": "influencerId",
    },
    "MarketRegime": {
        "description": "A detected market regime window (bull/bear/crisis/etc.)",
        "required_props": ["regimeId", "regimeType", "startTime", "endTime"],
        "optional_props": ["confidence", "avgVolatility", "trendDirection", "assetId"],
        "index_props": ["regimeId", "regimeType"],
        "unique_constraint": "regimeId",
    },
    "OrderBookLevel": {
        "description": "A specific price level in the limit order book",
        "required_props": ["levelId", "assetId", "side", "price", "volume", "timestamp"],
        "optional_props": ["isManipulation", "manipType", "imbalance"],
        "index_props": ["levelId", "assetId"],
        "unique_constraint": "levelId",
    },
    "Signal": {
        "description": "A detected manipulation signal from analytics",
        "required_props": ["signalId", "signalType", "assetId", "timestamp", "strength"],
        "optional_props": ["confidence", "features", "source", "isValidated"],
        "index_props": ["signalId", "signalType"],
        "unique_constraint": "signalId",
    },
    "RiskAlert": {
        "description": "An actionable manipulation risk alert",
        "required_props": ["alertId", "alertType", "severity", "riskScore", "timestamp"],
        "optional_props": ["description", "acknowledged", "assetId", "regulatoryFlag"],
        "index_props": ["alertId", "severity"],
        "unique_constraint": "alertId",
    },
    "Explanation": {
        "description": "XAI explanation for a model decision",
        "required_props": ["explanationId", "modelName", "decisionType", "confidence"],
        "optional_props": ["featureImportances", "counterfactual", "naturalLanguage", "alertId"],
        "index_props": ["explanationId"],
        "unique_constraint": "explanationId",
    },
}

RELATIONSHIP_TYPES = {
    "INFLUENCES": {
        "description": "Influencer affects asset price/sentiment",
        "source_types": ["Influencer"],
        "target_types": ["Asset"],
        "props": ["avgSentiment", "postCount", "weight", "timestamp"],
    },
    "CORRELATES_WITH": {
        "description": "Asset price co-movement correlation",
        "source_types": ["Asset"],
        "target_types": ["Asset"],
        "props": ["correlationCoeff", "pValue", "window", "absWeight"],
    },
    "HAS_REGIME": {
        "description": "Asset is in a specific market regime",
        "source_types": ["Asset"],
        "target_types": ["MarketRegime"],
        "props": ["startTime", "endTime", "confidence"],
    },
    "PROPAGATES_RISK": {
        "description": "Risk contagion between assets",
        "source_types": ["Asset", "Signal"],
        "target_types": ["Asset", "RiskAlert"],
        "props": ["riskWeight", "propagationDelay", "channel"],
    },
    "TRIGGERS_ALERT": {
        "description": "Signal triggers a risk alert",
        "source_types": ["Signal"],
        "target_types": ["RiskAlert"],
        "props": ["triggerStrength", "timestamp"],
    },
    "CAUSES_SIGNAL": {
        "description": "Asset/Regime/Influencer activity causes a signal",
        "source_types": ["Asset", "Influencer", "MarketRegime"],
        "target_types": ["Signal"],
        "props": ["causalStrength", "lag", "mechanism"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: GraphSchemaManager
# ──────────────────────────────────────────────────────────────────────────────

class GraphSchemaManager:
    """
    Manages the Neo4j graph schema for RAGNN.
    Generates Cypher DDL, validates schema consistency,
    and creates synthetic graph population scripts.
    """

    def __init__(self, module_name: str = "RAGNN"):
        self.module_name  = module_name
        self.node_types   = NODE_TYPES
        self.rel_types    = RELATIONSHIP_TYPES
        self._cypher_stmts: List[str] = []

    def generate_schema_cypher(self) -> str:
        """Generate CREATE CONSTRAINT and CREATE INDEX Cypher statements."""
        stmts = [
            "// ─────────────────────────────────────────────────────────────",
            f"// {self.module_name} Graph Schema",
            f"// Generated: {datetime.utcnow().isoformat()}Z",
            "// ─────────────────────────────────────────────────────────────",
            "",
        ]

        for label, info in self.node_types.items():
            stmts.append(f"// -- Node: {label} - {info['description']}")
            constraint = info.get("unique_constraint")
            if constraint:
                stmts.append(
                    f"CREATE CONSTRAINT {label.lower()}_unique_{constraint} "
                    f"IF NOT EXISTS FOR (n:{label}) REQUIRE n.{constraint} IS UNIQUE;"
                )
            for idx_prop in info.get("index_props", []):
                if idx_prop != constraint:
                    stmts.append(
                        f"CREATE INDEX {label.lower()}_{idx_prop}_idx "
                        f"IF NOT EXISTS FOR (n:{label}) ON (n.{idx_prop});"
                    )
            stmts.append("")

        self._cypher_stmts = stmts
        return "\n".join(stmts)

    def validate_schema(self) -> Tuple[bool, List[str]]:
        """Validate schema definitions for completeness and consistency."""
        errors = []

        for label, info in self.node_types.items():
            if "required_props" not in info:
                errors.append(f"Node {label} missing required_props")
            if "unique_constraint" not in info:
                errors.append(f"Node {label} missing unique_constraint")

        for rel, info in self.rel_types.items():
            if "source_types" not in info or "target_types" not in info:
                errors.append(f"Relationship {rel} missing source/target types")
            for stype in info.get("source_types", []):
                if stype not in self.node_types:
                    errors.append(f"Relationship {rel}: unknown source type {stype}")
            for ttype in info.get("target_types", []):
                if ttype not in self.node_types:
                    errors.append(f"Relationship {rel}: unknown target type {ttype}")

        return len(errors) == 0, errors

    def export_schema_json(self, filename: str = "schema_definition.json") -> str:
        """Export schema as JSON for documentation and tooling."""
        schema_doc = {
            "module": self.module_name,
            "full_name": "Regime-Aware Graph Neural Network",
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "node_types": self.node_types,
            "relationship_types": self.rel_types,
            "node_count": len(self.node_types),
            "relationship_count": len(self.rel_types),
        }
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(schema_doc, f, indent=2)
        print(f"  Schema JSON exported → {path}")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: SyntheticGraphPopulator
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticGraphPopulator:
    """
    Generates synthetic graph population scripts (Cypher INSERT statements)
    for testing and demonstrating the RAGNN schema without live Neo4j.
    """

    SECTORS    = ["Technology", "Finance", "Energy", "Healthcare",
                  "Consumer", "Utilities", "Materials", "Industrial"]
    EXCHANGES  = ["NYSE", "NASDAQ", "LSE", "TSX", "ASX", "HKEX"]
    PLATFORMS  = ["twitter", "reddit", "telegram", "discord", "stocktwits"]
    REGIMES    = ["bull", "bear", "sideways", "crisis", "recovery"]
    ALERT_TYPES = ["pump_and_dump", "spoofing", "layering", "wash_trading",
                   "momentum_ignition", "coordinated_sentiment", "quote_stuffing"]

    def __init__(self, n_assets: int = 20, n_influencers: int = 10):
        self.n_assets      = n_assets
        self.n_influencers = n_influencers
        self.assets        = [f"ASSET_{i:03d}" for i in range(n_assets)]
        self.influencers   = [f"INFL_{i:03d}" for i in range(n_influencers)]
        self._asset_props  = {}
        self._nodes: List[dict] = []
        self._edges: List[dict] = []

    def generate_asset_nodes(self) -> List[dict]:
        """Generate synthetic Asset node properties."""
        nodes = []
        for i, asset in enumerate(self.assets):
            props = {
                "assetId": asset,
                "symbol": f"SYM{i:03d}",
                "sector": self.SECTORS[i % len(self.SECTORS)],
                "exchange": self.EXCHANGES[i % len(self.EXCHANGES)],
                "listingDate": "2018-01-01",
                "currency": "USD",
                "marketCap": round(random.uniform(1e8, 1e12), 2),
                "manipulationRiskScore": round(random.uniform(0, 1), 4),
            }
            self._asset_props[asset] = props
            nodes.append({"label": "Asset", "props": props})
        self._nodes.extend(nodes)
        return nodes

    def generate_influencer_nodes(self) -> List[dict]:
        """Generate synthetic Influencer node properties."""
        nodes = []
        types = ["retail", "bot", "institutional", "whale", "analyst"]
        for i, infl in enumerate(self.influencers):
            props = {
                "influencerId": infl,
                "platform": self.PLATFORMS[i % len(self.PLATFORMS)],
                "followerCount": random.randint(100, 500000),
                "influencerType": types[i % len(types)],
                "avgSentiment": round(random.uniform(-0.5, 0.8), 4),
                "postCount": random.randint(10, 10000),
                "engagementRate": round(random.uniform(0.001, 0.05), 4),
                "isSuspicious": random.random() < 0.2,
            }
            nodes.append({"label": "Influencer", "props": props})
        self._nodes.extend(nodes)
        return nodes

    def generate_regime_nodes(self) -> List[dict]:
        """Generate synthetic MarketRegime nodes."""
        nodes = []
        base = datetime(2024, 1, 1)
        for i, asset in enumerate(self.assets[:10]):
            regime_type = self.REGIMES[i % len(self.REGIMES)]
            regime_id   = f"REGIME_{asset}_{i:03d}"
            props = {
                "regimeId": regime_id,
                "regimeType": regime_type,
                "startTime": (base + timedelta(days=i*5)).isoformat(),
                "endTime":   (base + timedelta(days=i*5+5)).isoformat(),
                "confidence": round(random.uniform(0.6, 0.99), 4),
                "avgVolatility": round(random.uniform(0.005, 0.04), 4),
                "trendDirection": random.choice(["up", "down", "flat"]),
                "assetId": asset,
            }
            nodes.append({"label": "MarketRegime", "props": props})
        self._nodes.extend(nodes)
        return nodes

    def generate_signal_nodes(self) -> List[dict]:
        """Generate synthetic Signal nodes."""
        nodes = []
        signal_types = ["price_spike", "volume_surge", "sentiment_burst",
                        "order_imbalance", "correlation_break"]
        for i in range(min(self.n_assets, 15)):
            asset = self.assets[i]
            for j, stype in enumerate(signal_types[:3]):
                sid = f"SIG_{asset}_{stype}_{j:03d}"
                props = {
                    "signalId": sid,
                    "signalType": stype,
                    "assetId": asset,
                    "timestamp": (datetime(2024, 3, 1) + timedelta(hours=i+j)).isoformat(),
                    "strength": round(random.uniform(0.3, 1.0), 4),
                    "confidence": round(random.uniform(0.5, 0.99), 4),
                    "source": "RAGNN",
                    "isValidated": random.random() < 0.7,
                }
                nodes.append({"label": "Signal", "props": props})
        self._nodes.extend(nodes)
        return nodes

    def generate_alert_nodes(self) -> List[dict]:
        """Generate synthetic RiskAlert nodes."""
        nodes = []
        severities = ["low", "medium", "high", "critical"]
        for i, asset in enumerate(self.assets[:8]):
            alert_type = self.ALERT_TYPES[i % len(self.ALERT_TYPES)]
            severity   = severities[min(i % 4, len(severities)-1)]
            props = {
                "alertId": f"ALERT_{asset}_{i:03d}",
                "alertType": alert_type,
                "severity": severity,
                "riskScore": round(random.uniform(0.4, 1.0), 4),
                "timestamp": (datetime(2024, 3, 15) + timedelta(hours=i)).isoformat(),
                "description": f"{alert_type.replace('_',' ').title()} detected in {asset}",
                "assetId": asset,
                "acknowledged": random.random() < 0.3,
                "regulatoryFlag": severity in ["high", "critical"],
            }
            nodes.append({"label": "RiskAlert", "props": props})
        self._nodes.extend(nodes)
        return nodes

    def generate_edges(self) -> List[dict]:
        """Generate synthetic graph edges."""
        edges = []
        # CORRELATES_WITH edges between assets
        for i in range(0, len(self.assets)-1, 2):
            edges.append({
                "source": self.assets[i],
                "target": self.assets[i+1],
                "relationship": "CORRELATES_WITH",
                "props": {
                    "correlationCoeff": round(random.uniform(-1, 1), 4),
                    "pValue": round(random.uniform(0, 0.05), 6),
                    "window": "20D",
                },
            })
        # INFLUENCES edges
        for infl in self.influencers[:5]:
            target = random.choice(self.assets)
            edges.append({
                "source": infl,
                "target": target,
                "relationship": "INFLUENCES",
                "props": {
                    "avgSentiment": round(random.uniform(-0.5, 0.8), 4),
                    "postCount": random.randint(5, 200),
                    "weight": round(random.uniform(0.1, 1.0), 4),
                },
            })
        self._edges.extend(edges)
        return edges

    def export_population_script(self, filename: str = "graph_population.json") -> str:
        path = os.path.join(OUTPUT_DIR, filename)
        data = {"nodes": self._nodes, "edges": self._edges,
                 "metadata": {"module": "RAGNN",
                               "generated_at": datetime.utcnow().isoformat()}}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Graph population script → {path} ({len(self._nodes)} nodes, {len(self._edges)} edges)")
        return path

    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        nodes_df = pd.DataFrame([
            {"id": n["props"].get(
                list(n["props"].keys())[0], f"node_{i}"),
             "label": n["label"], **n["props"]}
            for i, n in enumerate(self._nodes)
        ])
        edges_df = pd.DataFrame(self._edges)
        return nodes_df, edges_df


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: Neo4jMockExecutor
# ──────────────────────────────────────────────────────────────────────────────

class Neo4jMockExecutor:
    """
    Mock Neo4j executor for testing graph operations without a live database.
    Simulates query execution and returns realistic result structures.
    """

    def __init__(self):
        self.executed_queries: List[dict] = []
        self.node_store: Dict[str, dict] = {}
        self.edge_store: List[dict] = []

    def run(self, cypher: str, params: dict = None) -> List[dict]:
        """Mock query execution."""
        self.executed_queries.append({
            "query": cypher[:100] + "..." if len(cypher) > 100 else cypher,
            "params": params or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Return mock results based on query type
        if "MATCH" in cypher.upper() and "RETURN" in cypher.upper():
            return [{"n": {"id": "mock_node", "type": "Asset"}, "count": 42}]
        elif "CREATE" in cypher.upper():
            return [{"status": "created"}]
        elif "MERGE" in cypher.upper():
            return [{"status": "merged"}]
        return [{"status": "ok"}]

    def close(self):
        print(f"  MockNeo4j: {len(self.executed_queries)} queries executed.")

    def get_stats(self) -> dict:
        return {
            "total_queries": len(self.executed_queries),
            "nodes_in_store": len(self.node_store),
            "edges_in_store": len(self.edge_store),
        }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  RAGNN - Graph Schema Module")
    print(f"  Regime-Aware Graph Neural Network")
    print(f"{'='*60}\n")

    # Schema management
    schema_mgr = GraphSchemaManager("RAGNN")
    cypher      = schema_mgr.generate_schema_cypher()
    valid, errs = schema_mgr.validate_schema()
    print(f"[1/4] Schema validation: {'OK' if valid else 'FAILED'}")
    if errs:
        for e in errs:
            print(f"       ✗ {e}")
    schema_path = os.path.join(OUTPUT_DIR, "schema_cypher.txt")
    with open(schema_path, "w") as f:
        f.write(cypher)
    print(f"       Cypher DDL written → {schema_path}")
    schema_mgr.export_schema_json()

    # Graph population
    print("[2/4] Generating synthetic graph data...")
    populator = SyntheticGraphPopulator(n_assets=20, n_influencers=10)
    populator.generate_asset_nodes()
    populator.generate_influencer_nodes()
    populator.generate_regime_nodes()
    populator.generate_signal_nodes()
    populator.generate_alert_nodes()
    populator.generate_edges()
    populator.export_population_script()
    nodes_df, edges_df = populator.to_dataframes()
    print(f"       Nodes: {len(nodes_df)} | Edges: {len(edges_df)}")

    # Mock Neo4j execution
    print("[3/4] Running mock Neo4j queries...")
    mock_db = Neo4jMockExecutor()
    sample_queries = [
        "MATCH (a:Asset) RETURN a LIMIT 10",
        "CREATE (s:Signal {signalId: 'TEST_001', signalType: 'price_spike'})",
        "MATCH (a:Asset)-[:CORRELATES_WITH]->(b:Asset) RETURN a.assetId, b.assetId, r.correlationCoeff",
    ]
    for q in sample_queries:
        result = mock_db.run(q)
    mock_db.close()

    # Save summary
    print("[4/4] Saving graph schema summary...")
    summary = {
        "module": "RAGNN",
        "node_types": list(NODE_TYPES.keys()),
        "relationship_types": list(RELATIONSHIP_TYPES.keys()),
        "total_synthetic_nodes": len(nodes_df),
        "total_synthetic_edges": len(edges_df),
        "schema_valid": valid,
    }
    summary_path = os.path.join(OUTPUT_DIR, "schema_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Schema summary → {summary_path}")
    print(f"\nRAGNN Graph Schema Module Complete.\n")
    return nodes_df, edges_df

if __name__ == "__main__":
    main()
