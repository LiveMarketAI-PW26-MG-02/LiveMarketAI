#!/usr/bin/env python3
"""
Microstructure Graph Neural Network (MGNN) - Data Generation Module
Synthetic financial market data generation for manipulation detection research.
Implements regime-aware data synthesis, microstructure simulation,
sentiment generation, and graph-ready data preparation.
"""

import numpy as np
import pandas as pd
import json
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import networkx as nx
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

NUM_ASSETS        = 50
NUM_TRADING_DAYS  = 252
TICK_PER_DAY      = 390          # minutes in a trading day
MANIPULATION_PROB = 0.15         # 15% of windows flagged as manipulation
REGIME_TYPES      = ['normal_flow', 'spoofing', 'layering', 'quote_stuffing', 'momentum_ignition']
OUTPUT_DIR        = "outputs/mgnn_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CLASS: SyntheticMarketDataGenerator
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticMarketDataGenerator:
    """
    Generates synthetic OHLCV + microstructure + sentiment data for MGNN.
    Research motivation: realistic synthetic data allows controlled
    experimentation without data privacy constraints.
    """

    def __init__(self, n_assets: int = NUM_ASSETS, n_days: int = NUM_TRADING_DAYS):
        self.n_assets = n_assets
        self.n_days   = n_days
        self.assets   = [f"ASSET_{i:03d}" for i in range(n_assets)]
        self.scaler   = StandardScaler()
        self._regime_schedule = []
        self._prices = {}

    def generate_price_series(self, asset_id: str, regime: str = "bull") -> pd.DataFrame:
        """
        GBM-based price path with regime-specific drift and volatility.
        Regimes: ['normal_flow', 'spoofing', 'layering', 'quote_stuffing', 'momentum_ignition']
        """
        regime_params = {
            "bull":      {"mu": 0.0008, "sigma": 0.012},
            "bear":      {"mu": -0.0006, "sigma": 0.018},
            "sideways":  {"mu": 0.0001, "sigma": 0.008},
            "crisis":    {"mu": -0.003,  "sigma": 0.045},
            "recovery":  {"mu": 0.0015, "sigma": 0.020},
            "normal_flow": {"mu": 0.0005, "sigma": 0.010},
            "spoofing":  {"mu": 0.0002, "sigma": 0.025},
            "layering":  {"mu": 0.0003, "sigma": 0.022},
            "quote_stuffing": {"mu": 0.0001, "sigma": 0.030},
            "momentum_ignition": {"mu": 0.002, "sigma": 0.035},
            "organic_sentiment": {"mu": 0.0006, "sigma": 0.011},
            "coordinated_pump":  {"mu": 0.005,  "sigma": 0.040},
            "fud_campaign":      {"mu": -0.004,  "sigma": 0.038},
            "neutral":   {"mu": 0.0002, "sigma": 0.009},
            "mixed":     {"mu": 0.0003, "sigma": 0.015},
            "stable":    {"mu": 0.0004, "sigma": 0.007},
            "stress":    {"mu": -0.001,  "sigma": 0.028},
            "contagion": {"mu": -0.002,  "sigma": 0.042},
            "cascade":   {"mu": -0.005,  "sigma": 0.060},
            "tick_scale": {"mu": 0.00005,"sigma": 0.002},
            "high_confidence": {"mu": 0.0005,"sigma": 0.010},
            "uncertain": {"mu": 0.0002, "sigma": 0.020},
            "exploration": {"mu": 0.0003,"sigma": 0.012},
            "pre_manipulation": {"mu": 0.0002,"sigma": 0.008},
            "active_pump": {"mu": 0.006, "sigma": 0.050},
            "distribution": {"mu": 0.001, "sigma": 0.020},
            "dump":       {"mu": -0.008, "sigma": 0.065},
            "factual":   {"mu": 0.0005, "sigma": 0.010},
            "counterfactual_low": {"mu": 0.0001,"sigma": 0.005},
            "counterfactual_high": {"mu": 0.008, "sigma": 0.055},
        }
        params = regime_params.get(regime, {"mu": 0.0005, "sigma": 0.015})
        mu, sigma = params["mu"], params["sigma"]

        T = self.n_days * TICK_PER_DAY
        dt = 1.0 / TICK_PER_DAY
        S0 = random.uniform(10.0, 500.0)

        # Geometric Brownian Motion
        W = np.random.normal(0, np.sqrt(dt), T)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * W
        prices = S0 * np.exp(np.cumsum(log_returns))

        # Inject manipulation spikes
        manipulation_windows = []
        if random.random() < MANIPULATION_PROB:
            spike_idx = random.randint(T // 4, 3 * T // 4)
            spike_magnitude = random.uniform(0.05, 0.25)
            prices[spike_idx:spike_idx+30] *= (1 + spike_magnitude)
            prices[spike_idx+30:spike_idx+60] *= (1 - spike_magnitude * 0.6)
            manipulation_windows.append((spike_idx, spike_idx + 60))

        # Build OHLCV dataframe
        timestamps = pd.date_range("2024-01-01 09:30", periods=T, freq="1min")
        volume = np.random.lognormal(mean=10, sigma=1.5, size=T).astype(int)

        # High and Low derived from price with noise
        high = prices * (1 + np.abs(np.random.normal(0, 0.003, T)))
        low  = prices * (1 - np.abs(np.random.normal(0, 0.003, T)))
        open_p = np.roll(prices, 1); open_p[0] = S0

        df = pd.DataFrame({
            "timestamp": timestamps,
            "asset_id":  asset_id,
            "open":      open_p,
            "high":      high,
            "low":       low,
            "close":     prices,
            "volume":    volume,
            "regime":    regime,
            "is_manipulation": [
                any(s <= i <= e for s, e in manipulation_windows) for i in range(T)
            ],
        })
        self._prices[asset_id] = df
        return df

    def generate_all_assets(self) -> pd.DataFrame:
        """Generate price series for all assets across rotating regimes."""
        all_data = []
        for i, asset in enumerate(self.assets):
            # Rotate through regime types
            regime = REGIME_TYPES[i % len(REGIME_TYPES)]
            df = self.generate_price_series(asset, regime)
            all_data.append(df)
            if (i + 1) % 10 == 0:
                print(f"  Generated data for {i+1}/{self.n_assets} assets...")
        combined = pd.concat(all_data, ignore_index=True)
        return combined

    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log returns and rolling volatility features."""
        df = df.copy()
        df["log_return"]    = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["rolling_vol_5"] = df["log_return"].rolling(5).std().fillna(0)
        df["rolling_vol_20"]= df["log_return"].rolling(20).std().fillna(0)
        df["momentum_5"]    = df["close"].pct_change(5).fillna(0)
        df["momentum_20"]   = df["close"].pct_change(20).fillna(0)
        return df

    def save(self, df: pd.DataFrame, filename: str = "market_data.csv"):
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} rows → {path}")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: SentimentDataGenerator
# ──────────────────────────────────────────────────────────────────────────────

class SentimentDataGenerator:
    """
    Generates synthetic social media + news sentiment data aligned with
    price manipulation events for MGNN.
    """

    POSITIVE_WORDS = ["moon", "bullish", "pump", "surge", "buy", "breakout",
                      "rally", "explosive", "skyrocket", "massive gains"]
    NEGATIVE_WORDS = ["crash", "dump", "fraud", "scam", "sell", "collapse",
                      "plummet", "bearish", "warning", "exit"]
    NEUTRAL_WORDS  = ["analysis", "review", "update", "report", "volume",
                      "trading", "market", "position", "portfolio", "strategy"]

    def __init__(self, n_assets: int = NUM_ASSETS):
        self.n_assets = n_assets
        self.assets   = [f"ASSET_{i:03d}" for i in range(n_assets)]
        self.platforms = ["twitter", "reddit", "telegram", "discord", "stocktwits"]
        self.influencer_types = ["retail", "institutional", "bot", "whale", "analyst"]

    def generate_post(self, asset: str, sentiment_type: str, timestamp: datetime) -> dict:
        """Simulate a social media post with associated metadata."""
        if sentiment_type == "pump":
            words = random.choices(self.POSITIVE_WORDS, k=random.randint(3, 6))
            score = random.uniform(0.6, 1.0)
        elif sentiment_type == "dump":
            words = random.choices(self.NEGATIVE_WORDS, k=random.randint(3, 6))
            score = random.uniform(-1.0, -0.4)
        else:
            words = random.choices(self.NEUTRAL_WORDS, k=random.randint(2, 4))
            score = random.uniform(-0.2, 0.2)

        text = " ".join(words) + f" ${asset.replace('_', '')}"
        influencer_type = random.choice(self.influencer_types)
        follower_count  = self._follower_count(influencer_type)

        return {
            "timestamp":       timestamp.isoformat(),
            "asset_id":        asset,
            "platform":        random.choice(self.platforms),
            "text":            text,
            "sentiment_score": round(score, 4),
            "sentiment_type":  sentiment_type,
            "influencer_type": influencer_type,
            "follower_count":  follower_count,
            "engagement":      int(follower_count * random.uniform(0.001, 0.05)),
            "is_coordinated":  influencer_type == "bot" and abs(score) > 0.7,
        }

    def _follower_count(self, itype: str) -> int:
        mapping = {
            "retail": random.randint(10, 5000),
            "institutional": random.randint(50000, 500000),
            "bot": random.randint(100, 2000),
            "whale": random.randint(10000, 100000),
            "analyst": random.randint(5000, 50000),
        }
        return mapping.get(itype, 1000)

    def generate_dataset(self, n_posts_per_asset: int = 200) -> pd.DataFrame:
        """Generate full sentiment dataset across all assets."""
        records = []
        base_ts = datetime(2024, 1, 1, 9, 30)

        for asset in self.assets:
            # Randomly assign a manipulation event
            has_pump = random.random() < 0.2
            pump_start = random.randint(50, 150) if has_pump else None

            for i in range(n_posts_per_asset):
                ts = base_ts + timedelta(minutes=i * random.randint(1, 30))
                if has_pump and pump_start and pump_start <= i <= pump_start + 30:
                    stype = "pump"
                elif has_pump and pump_start and i > pump_start + 30:
                    stype = random.choice(["dump", "neutral"])
                else:
                    stype = random.choice(["neutral", "pump", "dump"],
                                         # weights
                                         ) if False else random.choices(
                        ["neutral", "pump", "dump"], weights=[0.7, 0.15, 0.15]
                    )[0]
                records.append(self.generate_post(asset, stype, ts))

        return pd.DataFrame(records)

    def save(self, df: pd.DataFrame, filename: str = "sentiment_data.csv"):
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} sentiment posts → {path}")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: OrderBookGenerator
# ──────────────────────────────────────────────────────────────────────────────

class OrderBookGenerator:
    """
    Simulates limit order book snapshots including spoofing and layering patterns.
    Critical for MGNN microstructure-based manipulation detection.
    """

    MANIPULATION_TYPES = ["none", "spoofing", "layering", "quote_stuffing",
                          "momentum_ignition", "wash_trading"]

    def __init__(self, n_levels: int = 10):
        self.n_levels = n_levels

    def generate_snapshot(self, mid_price: float,
                          manip_type: str = "none") -> dict:
        """Generate a single order book snapshot."""
        spread_frac = 0.0005 if manip_type == "none" else random.uniform(0.001, 0.003)
        spread = mid_price * spread_frac

        bid_prices = [mid_price - spread * (i + 1) for i in range(self.n_levels)]
        ask_prices = [mid_price + spread * (i + 1) for i in range(self.n_levels)]

        # Normal volumes
        bid_vols = np.random.exponential(scale=1000, size=self.n_levels).astype(int) + 100
        ask_vols = np.random.exponential(scale=1000, size=self.n_levels).astype(int) + 100

        if manip_type == "spoofing":
            # Large orders on one side, will be cancelled
            side = random.choice(["bid", "ask"])
            if side == "bid":
                bid_vols[0:3] *= random.randint(10, 50)
            else:
                ask_vols[0:3] *= random.randint(10, 50)

        elif manip_type == "layering":
            # Multiple layers at regular intervals
            for lvl in range(0, self.n_levels, 2):
                bid_vols[lvl] *= random.randint(5, 20)

        elif manip_type == "quote_stuffing":
            # Extremely high number of small orders flooding market
            bid_vols = np.ones(self.n_levels, dtype=int) * random.randint(1, 5)
            ask_vols = np.ones(self.n_levels, dtype=int) * random.randint(1, 5)

        return {
            "mid_price":   mid_price,
            "spread":      spread,
            "bid_prices":  bid_prices,
            "ask_prices":  ask_prices,
            "bid_volumes": bid_vols.tolist(),
            "ask_volumes": ask_vols.tolist(),
            "manip_type":  manip_type,
            "imbalance":   (sum(bid_vols) - sum(ask_vols)) / (sum(bid_vols) + sum(ask_vols)),
            "depth_total": int(sum(bid_vols) + sum(ask_vols)),
        }

    def generate_sequence(self, n_snapshots: int = 500,
                          asset_id: str = "ASSET_001") -> pd.DataFrame:
        """Generate time series of order book snapshots."""
        S0 = random.uniform(50, 200)
        prices = S0 * np.exp(np.cumsum(np.random.normal(0, 0.002, n_snapshots)))
        manip_start = random.randint(100, 300) if random.random() < 0.25 else None
        manip_type_chosen = random.choice(self.MANIPULATION_TYPES[1:])

        records = []
        for i, price in enumerate(prices):
            if manip_start and manip_start <= i < manip_start + 50:
                mtype = manip_type_chosen
            else:
                mtype = "none"
            snap = self.generate_snapshot(price, mtype)
            snap["timestamp"] = (datetime(2024, 1, 1, 9, 30) + timedelta(seconds=i)).isoformat()
            snap["asset_id"]  = asset_id
            snap["snapshot_id"] = i
            records.append(snap)

        return pd.DataFrame(records)

    def save(self, df: pd.DataFrame, filename: str = "orderbook_data.csv"):
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} order book snapshots → {path}")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: GraphDataBuilder
# ──────────────────────────────────────────────────────────────────────────────

class GraphDataBuilder:
    """
    Converts raw financial data into NetworkX graph format ready for Neo4j loading
    and GNN propagation in MGNN.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._node_counter = 0

    def build_asset_correlation_graph(self, price_df: pd.DataFrame,
                                      threshold: float = 0.5) -> nx.DiGraph:
        """
        Build asset correlation graph where edge weight > threshold
        indicates significant co-movement (potential manipulation link).
        """
        pivot = price_df.pivot_table(
            values="close", index="timestamp", columns="asset_id", aggfunc="last"
        ).ffill().bfill()

        corr_matrix = pivot.pct_change().dropna().corr()
        G = nx.DiGraph()

        # Add asset nodes
        for asset in corr_matrix.columns:
            G.add_node(asset, node_type="Asset",
                       label=asset, sector=self._assign_sector(asset))

        # Add correlation edges
        for i, a1 in enumerate(corr_matrix.columns):
            for j, a2 in enumerate(corr_matrix.columns):
                if i != j:
                    corr = corr_matrix.loc[a1, a2]
                    if abs(corr) > threshold:
                        G.add_edge(a1, a2,
                                   relationship="CORRELATES_WITH",
                                   weight=float(corr),
                                   abs_weight=float(abs(corr)))

        self.G = G
        print(f"  Correlation graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _assign_sector(self, asset_id: str) -> str:
        sectors = ["Technology", "Finance", "Energy", "Healthcare",
                   "Consumer", "Industrial", "Materials", "Utilities"]
        idx = int(asset_id.replace("ASSET_", "")) % len(sectors)
        return sectors[idx]

    def add_influencer_nodes(self, sentiment_df: pd.DataFrame) -> None:
        """Add influencer nodes and INFLUENCES edges to graph."""
        top_influencers = (
            sentiment_df.groupby("influencer_type")["engagement"]
            .sum().reset_index()
        )
        for _, row in top_influencers.iterrows():
            node_id = f"INFL_{row['influencer_type'].upper()}"
            self.G.add_node(node_id,
                            node_type="Influencer",
                            influencer_type=row["influencer_type"],
                            total_engagement=int(row["engagement"]))

        # Add influencer-asset edges
        asset_sentiment = (
            sentiment_df.groupby(["asset_id", "influencer_type"])["sentiment_score"]
            .mean().reset_index()
        )
        for _, row in asset_sentiment.iterrows():
            if row["asset_id"] in self.G.nodes:
                infl_node = f"INFL_{row['influencer_type'].upper()}"
                if infl_node in self.G.nodes:
                    self.G.add_edge(
                        infl_node, row["asset_id"],
                        relationship="INFLUENCES",
                        avg_sentiment=float(row["sentiment_score"]),
                    )

    def export_to_json(self, filename: str = "graph_data.json") -> str:
        """Export graph as JSON for Neo4j loading and GNN processing."""
        data = {
            "nodes": [
                {"id": n, **dict(self.G.nodes[n])}
                for n in self.G.nodes()
            ],
            "edges": [
                {"source": u, "target": v, **dict(d)}
                for u, v, d in self.G.edges(data=True)
            ],
            "stats": {
                "num_nodes": self.G.number_of_nodes(),
                "num_edges": self.G.number_of_edges(),
                "density":   nx.density(self.G),
                "is_directed": self.G.is_directed(),
            }
        }
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Graph exported → {path}")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n============================================================")
    print(f"  MGNN - Data Generation Pipeline")
    print(f"  Microstructure Graph Neural Network")
    print(f"============================================================\n")

    # Step 1: Generate price data
    print("[1/5] Generating synthetic market price data...")
    price_gen = SyntheticMarketDataGenerator(n_assets=NUM_ASSETS, n_days=NUM_TRADING_DAYS)
    price_df  = price_gen.generate_all_assets()
    price_df  = price_gen.compute_returns(price_df)
    price_gen.save(price_df, "market_data.csv")
    print(f"      → {len(price_df):,} rows × {len(price_df.columns)} columns")

    # Step 2: Generate sentiment data
    print("[2/5] Generating synthetic sentiment data...")
    sent_gen = SentimentDataGenerator(n_assets=NUM_ASSETS)
    sent_df  = sent_gen.generate_dataset(n_posts_per_asset=200)
    sent_gen.save(sent_df, "sentiment_data.csv")

    # Step 3: Generate order book data
    print("[3/5] Generating order book snapshots...")
    ob_gen = OrderBookGenerator(n_levels=10)
    ob_records = []
    for asset in price_gen.assets[:10]:  # sample 10 assets for orderbook
        ob_df = ob_gen.generate_sequence(n_snapshots=200, asset_id=asset)
        ob_records.append(ob_df)
    ob_combined = pd.concat(ob_records, ignore_index=True)
    ob_gen.save(ob_combined, "orderbook_data.csv")

    # Step 4: Build graph structure
    print("[4/5] Building correlation graph...")
    # Use a sample for speed
    sample_df = price_df[price_df["asset_id"].isin(price_gen.assets[:20])].copy()
    sample_df = sample_df.drop_duplicates(subset=["timestamp", "asset_id"])
    graph_builder = GraphDataBuilder()
    graph_builder.build_asset_correlation_graph(sample_df, threshold=0.4)
    graph_builder.add_influencer_nodes(sent_df)
    graph_builder.export_to_json("graph_data.json")

    # Step 5: Generate manipulation labels
    print("[5/5] Generating manipulation label summary...")
    manipulation_summary = price_df.groupby("asset_id")["is_manipulation"].mean()
    label_df = manipulation_summary.reset_index()
    label_df.columns = ["asset_id", "manipulation_rate"]
    label_df["label"] = (label_df["manipulation_rate"] > 0).astype(int)
    label_path = os.path.join(OUTPUT_DIR, "manipulation_labels.csv")
    label_df.to_csv(label_path, index=False)
    print(f"      → {label_df['label'].sum()} / {len(label_df)} assets flagged")

    # Summary
    print(f"\n============================================================")
    print(f"  Data Generation Complete for MGNN")
    print(f"  Regime Types Simulated: {REGIME_TYPES}")
    print(f"  Assets: {NUM_ASSETS} | Days: {NUM_TRADING_DAYS}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"============================================================\n")
    return price_df, sent_df, ob_combined

if __name__ == "__main__":
    main()
