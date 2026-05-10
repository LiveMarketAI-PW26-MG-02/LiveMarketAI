// ═══════════════════════════════════════════════════════════════════════════
// Regime-Aware Graph Neural Network (RAGNN) — Operational Neo4j Queries
// ═══════════════════════════════════════════════════════════════════════════

// ── Q1: High-risk assets ──────────────────────────────────────────────────
// Find all assets with manipulation risk above threshold
MATCH (a:Asset)
WHERE a.manipulationRiskScore >= 0.7
RETURN a.assetId AS asset, a.symbol AS symbol,
       a.sector AS sector, a.manipulationRiskScore AS riskScore,
       a.exchange AS exchange
ORDER BY a.manipulationRiskScore DESC
LIMIT 20;

// ── Q2: Active critical alerts ────────────────────────────────────────────
MATCH (al:RiskAlert)
WHERE al.severity IN ['high', 'critical']
  AND al.acknowledged = false
  AND al.module = 'RAGNN'
RETURN al.alertId, al.alertType, al.severity, al.riskScore,
       al.timestamp, al.assetId, al.description
ORDER BY al.riskScore DESC;

// ── Q3: Influencer → Asset manipulation chain ─────────────────────────────
MATCH (i:Influencer)-[r1:INFLUENCES]->(a:Asset)<-[r2:CAUSES_SIGNAL]-(a)
-[:CAUSES_SIGNAL]->(s:Signal)-[:TRIGGERS_ALERT]->(al:RiskAlert)
WHERE i.isSuspicious = true
  AND al.severity IN ['high', 'critical']
RETURN i.influencerId, i.influencerType, i.platform,
       a.assetId, s.signalType, s.strength,
       al.alertType, al.riskScore
ORDER BY al.riskScore DESC;

// ── Q4: Regime-based risk clustering ─────────────────────────────────────
MATCH (a:Asset)-[:HAS_REGIME]->(r:MarketRegime)
WHERE r.regimeType IN ['crisis', 'bear']
WITH r.regimeType AS regime,
     collect(a.assetId) AS assets,
     avg(a.manipulationRiskScore) AS avgRisk,
     count(a) AS n
RETURN regime, n, avgRisk, assets[..5] AS sampleAssets
ORDER BY avgRisk DESC;

// ── Q5: Correlated high-risk asset pairs ─────────────────────────────────
MATCH (a1:Asset)-[c:CORRELATES_WITH]->(a2:Asset)
WHERE abs(c.correlationCoeff) > 0.7
  AND a1.manipulationRiskScore > 0.6
  AND a2.manipulationRiskScore > 0.6
RETURN a1.assetId AS asset1, a2.assetId AS asset2,
       c.correlationCoeff AS correlation,
       a1.manipulationRiskScore AS risk1,
       a2.manipulationRiskScore AS risk2
ORDER BY (a1.manipulationRiskScore + a2.manipulationRiskScore) DESC
LIMIT 15;

// ── Q6: Risk propagation paths ────────────────────────────────────────────
MATCH path = (a1:Asset)-[:PROPAGATES_RISK*1..3]->(a2:Asset)
WHERE a1.manipulationRiskScore > 0.8
RETURN a1.assetId AS source,
       [node IN nodes(path) | node.assetId] AS propagation_path,
       length(path) AS hops,
       a2.assetId AS destination,
       a2.manipulationRiskScore AS destRisk
ORDER BY a2.manipulationRiskScore DESC
LIMIT 10;

// ── Q7: Signal timeline for an asset ────────────────────────────────────
MATCH (a:Asset {assetId: 'ASSET_DEMO_001'})-[:CAUSES_SIGNAL]->(s:Signal)
OPTIONAL MATCH (s)-[:TRIGGERS_ALERT]->(al:RiskAlert)
RETURN s.signalId, s.signalType, s.timestamp, s.strength, s.confidence,
       al.alertId AS alert, al.severity AS alertSeverity
ORDER BY s.timestamp;

// ── Q8: Suspicious influencer network ────────────────────────────────────
MATCH (i:Influencer)-[:INFLUENCES]->(a:Asset)
WHERE i.isSuspicious = true
WITH i, collect(DISTINCT a.assetId) AS targetAssets,
     count(DISTINCT a) AS nTargets,
     avg(a.manipulationRiskScore) AS avgTargetRisk
WHERE nTargets >= 2
RETURN i.influencerId, i.influencerType, i.platform,
       i.followerCount, nTargets, avgTargetRisk, targetAssets
ORDER BY avgTargetRisk DESC;

// ── Q9: Top explanations for audit ───────────────────────────────────────
MATCH (al:RiskAlert)-[:HAS_EXPLANATION]->(ex:Explanation)
WHERE al.regulatoryFlag = true
RETURN al.alertId, al.alertType, al.riskScore,
       ex.explanationId, ex.modelName, ex.confidence,
       ex.naturalLanguage
ORDER BY al.riskScore DESC
LIMIT 10;

// ── Q10: Module performance summary ──────────────────────────────────────
MATCH (al:RiskAlert)
WHERE al.module = 'RAGNN'
WITH al.severity AS sev,
     count(*) AS n,
     avg(al.riskScore) AS avgScore,
     sum(CASE WHEN al.acknowledged THEN 1 ELSE 0 END) AS ackCount
RETURN sev, n, avgScore, ackCount,
       toFloat(ackCount)/n AS ackRate
ORDER BY avgScore DESC;

// ── Q11: Graph centrality analysis ───────────────────────────────────────
MATCH (a:Asset)
OPTIONAL MATCH (a)-[r:CORRELATES_WITH]->()
WITH a, count(r) AS outDegree
OPTIONAL MATCH ()-[r2:CORRELATES_WITH]->(a)
WITH a, outDegree, count(r2) AS inDegree
RETURN a.assetId, a.sector,
       outDegree + inDegree AS totalDegree,
       a.manipulationRiskScore AS riskScore
ORDER BY totalDegree DESC
LIMIT 15;

// ── Q12: Time-windowed alert density ─────────────────────────────────────
MATCH (al:RiskAlert)
WHERE al.module = 'RAGNN'
  AND al.timestamp >= '2024-01-01T00:00:00'
WITH substring(al.timestamp, 0, 7) AS month,
     al.severity AS severity,
     count(*) AS alertCount
RETURN month, severity, alertCount
ORDER BY month, alertCount DESC;

// ── Q13: Cross-module risk comparison (if multiple modules loaded) ────────
MATCH (al:RiskAlert)
WITH al.module AS module,
     avg(al.riskScore) AS avgRisk,
     count(*) AS totalAlerts,
     sum(CASE WHEN al.severity = 'critical' THEN 1 ELSE 0 END) AS criticalCount
RETURN module, totalAlerts, avgRisk, criticalCount
ORDER BY avgRisk DESC;
