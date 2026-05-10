#!/usr/bin/env python3
"""
Regime-Aware Graph Neural Network (RAGNN) - Feature Engineering Engine
Comprehensive feature pipeline: technical indicators, microstructure features,
sentiment aggregation, graph topology features, and cross-asset features.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

OUTPUT_DIR = "outputs/ragnn_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

class TechnicalIndicators:
    """OHLCV-based technical indicators for manipulation signal extraction."""

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(returns: pd.Series, period: int = 14) -> pd.Series:
        gain = returns.clip(lower=0).rolling(period).mean()
        loss = (-returns.clip(upper=0)).rolling(period).mean()
        rs   = gain / (loss + 1e-10)
        return (100 - 100 / (1 + rs)).fillna(50)

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
        ema_fast    = TechnicalIndicators.ema(close, fast)
        ema_slow    = TechnicalIndicators.ema(close, slow)
        macd_line   = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram   = macd_line - signal_line
        return pd.DataFrame({
            "macd": macd_line, "signal": signal_line, "histogram": histogram
        })

    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20,
                         n_std: float = 2.0) -> pd.DataFrame:
        ma    = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = ma + n_std * std
        lower = ma - n_std * std
        pct_b = (close - lower) / (upper - lower + 1e-10)
        return pd.DataFrame({"bb_upper": upper, "bb_lower": lower,
                               "bb_mid": ma, "bb_pct_b": pct_b})

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
        return (close * volume).cumsum() / volume.cumsum().replace(0, 1)

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "close" not in df.columns:
            return df
        close  = df["close"]
        volume = df.get("volume", pd.Series(1000, index=df.index))
        high   = df.get("high", close * 1.001)
        low    = df.get("low",  close * 0.999)

        df["ema_9"]  = self.ema(close, 9)
        df["ema_21"] = self.ema(close, 21)
        df["rsi_14"] = self.rsi(close.pct_change().fillna(0))

        macd_df = self.macd(close)
        df["macd"]          = macd_df["macd"]
        df["macd_signal"]   = macd_df["signal"]
        df["macd_histogram"]= macd_df["histogram"]

        bb_df = self.bollinger_bands(close)
        df["bb_pct_b"]  = bb_df["bb_pct_b"]
        df["bb_upper"]  = bb_df["bb_upper"]
        df["bb_lower"]  = bb_df["bb_lower"]

        df["atr_14"]  = self.atr(high, low, close)
        df["obv"]     = self.obv(close, volume)
        df["vwap"]    = self.vwap(close, volume)
        df["vwap_dev"]= (close - df["vwap"]) / (df["vwap"] + 1e-10)

        # Momentum
        for p in [5, 10, 20]:
            df[f"mom_{p}"]  = close.pct_change(p).fillna(0)
            df[f"vol_{p}"]  = close.pct_change().rolling(p).std().fillna(0)

        return df.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# MICROSTRUCTURE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class MicrostructureFeatureExtractor:
    """Extracts microstructure-based features from order book data."""

    def compute_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "bid_ask_spread" not in df.columns:
            df["bid_ask_spread"] = np.abs(np.random.randn(len(df))) * 0.01
        spread = df["bid_ask_spread"]
        df["spread_ma_5"]   = spread.rolling(5).mean().fillna(spread)
        df["spread_zscore"] = ((spread - spread.rolling(20).mean()) /
                               (spread.rolling(20).std() + 1e-10)).fillna(0)
        df["spread_spike"]  = (df["spread_zscore"].abs() > 2.5).astype(int)
        return df

    def compute_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "order_imbalance" not in df.columns:
            df["order_imbalance"] = np.random.uniform(-0.5, 0.5, len(df))
        imb = df["order_imbalance"]
        df["imbalance_ma_5"]   = imb.rolling(5).mean().fillna(0)
        df["imbalance_extreme"] = (imb.abs() > 0.7).astype(int)
        df["imbalance_flip"]   = (imb * imb.shift(1).fillna(1) < 0).astype(int)
        return df

    def compute_cancellation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "cancel_rate" not in df.columns:
            df["cancel_rate"] = np.random.beta(2, 10, len(df))
        cr = df["cancel_rate"]
        df["cancel_spike"] = (cr > cr.quantile(0.9)).astype(int)
        df["cancel_ma_10"] = cr.rolling(10).mean().fillna(cr)
        return df

    def compute_trade_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "volume" in df.columns:
            vol = df["volume"]
            df["trade_intensity"] = vol / (vol.rolling(20).mean() + 1)
        else:
            df["trade_intensity"] = np.random.lognormal(0, 0.5, len(df))
        return df

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_spread_features(df)
        df = self.compute_imbalance_features(df)
        df = self.compute_cancellation_features(df)
        df = self.compute_trade_intensity(df)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT AGGREGATION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class SentimentFeatureAggregator:
    """Aggregates raw sentiment posts into asset-level features."""

    def aggregate_1h(self, sent_df: pd.DataFrame,
                      price_df: pd.DataFrame) -> pd.DataFrame:
        sent_agg = sent_df.groupby("asset_id").agg(
            avg_sent        = ("sentiment_score", "mean"),
            sent_std        = ("sentiment_score", "std"),
            coord_ratio     = ("is_coordinated", "mean"),
            n_posts         = ("sentiment_score", "count"),
            max_engagement  = ("engagement", "max"),
            bot_ratio       = ("influencer_type", lambda x: (x == "bot").mean()),
        ).reset_index().fillna(0)
        return price_df.merge(sent_agg, on="asset_id", how="left").fillna(0)

    def sentiment_velocity(self, sent_df: pd.DataFrame,
                            window: str = "1H") -> pd.DataFrame:
        try:
            sent_df["ts"] = pd.to_datetime(sent_df["timestamp"])
            hourly = sent_df.groupby(["asset_id", pd.Grouper(key="ts", freq=window)])[
                "sentiment_score"
            ].mean().reset_index()
            hourly["sent_velocity"] = hourly.groupby("asset_id")["sentiment_score"].diff().fillna(0)
            return hourly
        except Exception:
            return sent_df

    def compute_all(self, sent_df: pd.DataFrame,
                    price_df: pd.DataFrame) -> pd.DataFrame:
        merged = self.aggregate_1h(sent_df, price_df)
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class GraphFeatureExtractor:
    """Computes graph topology features using networkx."""

    def extract(self, G) -> pd.DataFrame:
        import networkx as nx
        try:
            deg_cent = nx.degree_centrality(G)
        except Exception:
            deg_cent = {n: 0.0 for n in G.nodes()}
        try:
            bet_cent = nx.betweenness_centrality(G, normalized=True)
        except Exception:
            bet_cent = {n: 0.0 for n in G.nodes()}
        try:
            clos_cent = nx.closeness_centrality(G)
        except Exception:
            clos_cent = {n: 0.0 for n in G.nodes()}
        try:
            pr = nx.pagerank(G, alpha=0.85)
        except Exception:
            n = G.number_of_nodes()
            pr = {node: 1.0/n for node in G.nodes()}
        try:
            clust = nx.clustering(G.to_undirected())
        except Exception:
            clust = {n: 0.0 for n in G.nodes()}
        try:
            eig = nx.eigenvector_centrality(G, max_iter=100)
        except Exception:
            eig = {n: 0.0 for n in G.nodes()}

        rows = []
        for node in G.nodes():
            rows.append({
                "asset_id":               node,
                "degree_centrality":      deg_cent.get(node, 0),
                "betweenness_centrality": bet_cent.get(node, 0),
                "closeness_centrality":   clos_cent.get(node, 0),
                "pagerank":               pr.get(node, 0),
                "clustering_coeff":       clust.get(node, 0),
                "eigenvector_centrality": eig.get(node, 0),
                "in_degree":              G.in_degree(node) if G.is_directed() else G.degree(node),
                "out_degree":             G.out_degree(node) if G.is_directed() else G.degree(node),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePipelineOrchestrator:
    """Orchestrates the complete feature engineering pipeline for RAGNN."""

    def __init__(self):
        self.tech_ind   = TechnicalIndicators()
        self.micro_feat = MicrostructureFeatureExtractor()
        self.sent_feat  = SentimentFeatureAggregator()
        self.graph_feat = GraphFeatureExtractor()

    def run(self, price_df: pd.DataFrame,
            sent_df: Optional[pd.DataFrame] = None,
            G=None) -> pd.DataFrame:
        print(f"  [RAGNN] Running feature pipeline on {len(price_df)} rows...")

        enriched = self.tech_ind.compute_all(price_df)
        enriched = self.micro_feat.compute_all(enriched)

        if sent_df is not None and len(sent_df) > 0:
            enriched = self.sent_feat.compute_all(sent_df, enriched)

        if G is not None:
            graph_feats = self.graph_feat.extract(G)
            enriched    = enriched.merge(graph_feats, on="asset_id", how="left")
            for col in ["degree_centrality","betweenness_centrality","pagerank"]:
                if col in enriched.columns:
                    enriched[col].fillna(0, inplace=True)

        print(f"  [RAGNN] Feature matrix: {enriched.shape}")
        return enriched

    def save_features(self, df: pd.DataFrame,
                      filename: str = "features.csv") -> str:
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  Features saved → {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nRAGNN Feature Engineering Engine")

    # Generate minimal synthetic data for demo
    import networkx as nx
    n_assets = 15
    assets   = [f"ASSET_{i:03d}" for i in range(n_assets)]

    price_df = pd.DataFrame({
        "asset_id": assets,
        "close":    np.random.uniform(50, 200, n_assets),
        "high":     np.random.uniform(51, 205, n_assets),
        "low":      np.random.uniform(49, 195, n_assets),
        "volume":   np.random.randint(1000, 50000, n_assets),
        "is_manipulation": (np.random.rand(n_assets) < 0.15).astype(int),
    })
    sent_df = pd.DataFrame({
        "asset_id":         np.random.choice(assets, 100),
        "sentiment_score":  np.random.uniform(-0.5, 0.8, 100),
        "is_coordinated":   (np.random.rand(100) < 0.1).astype(bool),
        "engagement":       np.random.randint(10, 5000, 100),
        "influencer_type":  np.random.choice(["retail","bot","analyst"], 100),
        "timestamp":        [datetime.utcnow().isoformat() for _ in range(100)],
    })
    G = nx.erdos_renyi_graph(n_assets, 0.3, directed=True)
    mapping = {i: assets[i] for i in range(n_assets)}
    G = nx.relabel_nodes(G, mapping)

    pipeline = FeaturePipelineOrchestrator()
    feat_df  = pipeline.run(price_df, sent_df, G)
    pipeline.save_features(feat_df)

    print(f"  Feature columns: {len(feat_df.columns)}")
    print(f"  Shape: {feat_df.shape}")
    print(f"\nRAGNN Feature Engineering complete.\n")
    return feat_df

if __name__ == "__main__":
    main()
