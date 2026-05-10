// ═══════════════════════════════════════════════════════════════════════════
// Uncertainty-Aware Graph Neural Network (UGNN) — Neo4j Graph Schema
// Module: ugnn
// ═══════════════════════════════════════════════════════════════════════════

// ── CONSTRAINTS ─────────────────────────────────────────────────────────────

CREATE CONSTRAINT asset_unique_id IF NOT EXISTS
  FOR (n:Asset) REQUIRE n.assetId IS UNIQUE;

CREATE CONSTRAINT influencer_unique_id IF NOT EXISTS
  FOR (n:Influencer) REQUIRE n.influencerId IS UNIQUE;

CREATE CONSTRAINT regime_unique_id IF NOT EXISTS
  FOR (n:MarketRegime) REQUIRE n.regimeId IS UNIQUE;

CREATE CONSTRAINT orderbook_unique_id IF NOT EXISTS
  FOR (n:OrderBookLevel) REQUIRE n.levelId IS UNIQUE;

CREATE CONSTRAINT signal_unique_id IF NOT EXISTS
  FOR (n:Signal) REQUIRE n.signalId IS UNIQUE;

CREATE CONSTRAINT alert_unique_id IF NOT EXISTS
  FOR (n:RiskAlert) REQUIRE n.alertId IS UNIQUE;

CREATE CONSTRAINT explanation_unique_id IF NOT EXISTS
  FOR (n:Explanation) REQUIRE n.explanationId IS UNIQUE;

// ── INDEXES ──────────────────────────────────────────────────────────────────

CREATE INDEX asset_symbol_idx IF NOT EXISTS FOR (n:Asset) ON (n.symbol);
CREATE INDEX asset_sector_idx IF NOT EXISTS FOR (n:Asset) ON (n.sector);
CREATE INDEX asset_exchange_idx IF NOT EXISTS FOR (n:Asset) ON (n.exchange);
CREATE INDEX asset_risk_idx IF NOT EXISTS FOR (n:Asset) ON (n.manipulationRiskScore);

CREATE INDEX influencer_type_idx IF NOT EXISTS FOR (n:Influencer) ON (n.influencerType);
CREATE INDEX influencer_platform_idx IF NOT EXISTS FOR (n:Influencer) ON (n.platform);
CREATE INDEX influencer_suspicious_idx IF NOT EXISTS FOR (n:Influencer) ON (n.isSuspicious);

CREATE INDEX regime_type_idx IF NOT EXISTS FOR (n:MarketRegime) ON (n.regimeType);
CREATE INDEX regime_start_idx IF NOT EXISTS FOR (n:MarketRegime) ON (n.startTime);
CREATE INDEX regime_asset_idx IF NOT EXISTS FOR (n:MarketRegime) ON (n.assetId);

CREATE INDEX signal_type_idx IF NOT EXISTS FOR (n:Signal) ON (n.signalType);
CREATE INDEX signal_ts_idx IF NOT EXISTS FOR (n:Signal) ON (n.timestamp);
CREATE INDEX signal_asset_idx IF NOT EXISTS FOR (n:Signal) ON (n.assetId);
CREATE INDEX signal_strength_idx IF NOT EXISTS FOR (n:Signal) ON (n.strength);

CREATE INDEX alert_severity_idx IF NOT EXISTS FOR (n:RiskAlert) ON (n.severity);
CREATE INDEX alert_type_idx IF NOT EXISTS FOR (n:RiskAlert) ON (n.alertType);
CREATE INDEX alert_ts_idx IF NOT EXISTS FOR (n:RiskAlert) ON (n.timestamp);
CREATE INDEX alert_score_idx IF NOT EXISTS FOR (n:RiskAlert) ON (n.riskScore);
CREATE INDEX alert_ack_idx IF NOT EXISTS FOR (n:RiskAlert) ON (n.acknowledged);

CREATE INDEX explanation_model_idx IF NOT EXISTS FOR (n:Explanation) ON (n.modelName);

// ── NODE PROPERTIES (sample MERGE for UGNN) ─────────────────────────────

MERGE (a:Asset {assetId: 'ASSET_DEMO_001'})
ON CREATE SET
  a.symbol               = 'DEMO',
  a.sector               = 'Technology',
  a.exchange             = 'NASDAQ',
  a.marketCapTier        = 'large',
  a.listingDate          = '2018-01-01',
  a.currency             = 'USD',
  a.manipulationRiskScore = 0.0,
  a.module               = 'UGNN',
  a.createdAt            = datetime();

MERGE (infl:Influencer {influencerId: 'INFL_DEMO_001'})
ON CREATE SET
  infl.platform         = 'twitter',
  infl.followerCount    = 50000,
  infl.influencerType   = 'whale',
  infl.avgSentiment     = 0.6,
  infl.postCount        = 1200,
  infl.engagementRate   = 0.025,
  infl.isSuspicious     = false,
  infl.module           = 'UGNN',
  infl.createdAt        = datetime();

MERGE (r:MarketRegime {regimeId: 'REGIME_DEMO_001'})
ON CREATE SET
  r.regimeType       = 'bull',
  r.startTime        = '2024-01-01T09:30:00',
  r.endTime          = '2024-03-31T16:00:00',
  r.confidence       = 0.87,
  r.avgVolatility    = 0.012,
  r.trendDirection   = 'up',
  r.assetId          = 'ASSET_DEMO_001',
  r.module           = 'UGNN',
  r.createdAt        = datetime();

MERGE (sig:Signal {signalId: 'SIG_DEMO_001'})
ON CREATE SET
  sig.signalType   = 'price_spike',
  sig.assetId      = 'ASSET_DEMO_001',
  sig.timestamp    = '2024-03-15T14:30:00',
  sig.strength     = 0.82,
  sig.confidence   = 0.79,
  sig.source       = 'UGNN',
  sig.isValidated  = true,
  sig.module       = 'UGNN',
  sig.createdAt    = datetime();

MERGE (al:RiskAlert {alertId: 'ALERT_DEMO_001'})
ON CREATE SET
  al.alertType      = 'pump_and_dump',
  al.severity       = 'high',
  al.riskScore      = 0.83,
  al.timestamp      = '2024-03-15T14:31:00',
  al.description    = 'Pump-and-dump pattern detected by UGNN',
  al.assetId        = 'ASSET_DEMO_001',
  al.acknowledged   = false,
  al.regulatoryFlag = true,
  al.module         = 'UGNN',
  al.createdAt      = datetime();

MERGE (ex:Explanation {explanationId: 'EXP_DEMO_001'})
ON CREATE SET
  ex.modelName        = 'UGNN',
  ex.decisionType     = 'manipulation_flag',
  ex.confidence       = 0.83,
  ex.naturalLanguage  = 'Pump-and-dump pattern identified via regime-aware GNN',
  ex.alertId          = 'ALERT_DEMO_001',
  ex.module           = 'UGNN',
  ex.createdAt        = datetime();

// ── RELATIONSHIPS ────────────────────────────────────────────────────────────

MATCH (a:Asset {assetId: 'ASSET_DEMO_001'}),
      (r:MarketRegime {regimeId: 'REGIME_DEMO_001'})
MERGE (a)-[rel:HAS_REGIME]->(r)
ON CREATE SET
  rel.startTime  = '2024-01-01T09:30:00',
  rel.endTime    = '2024-03-31T16:00:00',
  rel.confidence = 0.87;

MATCH (i:Influencer {influencerId: 'INFL_DEMO_001'}),
      (a:Asset {assetId: 'ASSET_DEMO_001'})
MERGE (i)-[rel:INFLUENCES]->(a)
ON CREATE SET
  rel.avgSentiment = 0.6,
  rel.postCount    = 1200,
  rel.weight       = 0.72;

MATCH (sig:Signal {signalId: 'SIG_DEMO_001'}),
      (al:RiskAlert {alertId: 'ALERT_DEMO_001'})
MERGE (sig)-[rel:TRIGGERS_ALERT]->(al)
ON CREATE SET
  rel.triggerStrength = 0.82,
  rel.timestamp       = '2024-03-15T14:30:00';

MATCH (a:Asset {assetId: 'ASSET_DEMO_001'}),
      (sig:Signal {signalId: 'SIG_DEMO_001'})
MERGE (a)-[rel:CAUSES_SIGNAL]->(sig)
ON CREATE SET
  rel.causalStrength = 0.85,
  rel.lag            = 0,
  rel.mechanism      = 'price_volume_anomaly';

MATCH (al:RiskAlert {alertId: 'ALERT_DEMO_001'}),
      (ex:Explanation {explanationId: 'EXP_DEMO_001'})
MERGE (al)-[rel:HAS_EXPLANATION]->(ex)
ON CREATE SET rel.generatedAt = datetime();
