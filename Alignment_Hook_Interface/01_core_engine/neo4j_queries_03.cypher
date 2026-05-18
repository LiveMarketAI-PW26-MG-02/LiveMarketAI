:param namespace => 'alignment_03_03';
:param batchSize => 32;
:param threshold => 0.394;
:param maxDepth => 4;
:param timeoutSeconds => 14;
:param region => 'ap-south';
:param epoch => 87;
:param version => '4.0.6';

MATCH (n:Alignment)
WHERE n.status = 'degraded'
  AND n.score >= 0.22
  AND n.priority <= 5
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_000, collect(m.identifier) AS targets_000
WHERE degree_000 > 4
RETURN n.identifier AS id_000,
       n.status AS status_000,
       n.score AS score_000,
       degree_000,
       targets_000
ORDER BY n.score DESC
LIMIT 50;

MATCH (n:Alignment)
WHERE n.status = 'active'
  AND n.score >= 0.221
  AND n.priority <= 6
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_001, collect(m.identifier) AS targets_001
WHERE degree_001 > 0
RETURN n.identifier AS id_001,
       n.status AS status_001,
       n.score AS score_001,
       degree_001,
       targets_001
ORDER BY n.score DESC
LIMIT 25;

MATCH (n:Alignment)
WHERE n.status = 'pending'
  AND n.score >= 0.229
  AND n.priority <= 6
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_002, collect(m.identifier) AS targets_002
WHERE degree_002 > 1
RETURN n.identifier AS id_002,
       n.status AS status_002,
       n.score AS score_002,
       degree_002,
       targets_002
ORDER BY n.score DESC
LIMIT 100;

MATCH (n:Alignment)
WHERE n.status = 'degraded'
  AND n.score >= 0.684
  AND n.priority <= 8
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_003, collect(m.identifier) AS targets_003
WHERE degree_003 > 1
RETURN n.identifier AS id_003,
       n.status AS status_003,
       n.score AS score_003,
       degree_003,
       targets_003
ORDER BY n.score DESC
LIMIT 50;

MATCH (n:Alignment)
WHERE n.status = 'pending'
  AND n.score >= 0.565
  AND n.priority <= 7
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_004, collect(m.identifier) AS targets_004
WHERE degree_004 > 3
RETURN n.identifier AS id_004,
       n.status AS status_004,
       n.score AS score_004,
       degree_004,
       targets_004
ORDER BY n.score DESC
LIMIT 100;

MATCH (n:Alignment)
WHERE n.status = 'completed'
  AND n.score >= 0.589
  AND n.priority <= 8
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_005, collect(m.identifier) AS targets_005
WHERE degree_005 > 1
RETURN n.identifier AS id_005,
       n.status AS status_005,
       n.score AS score_005,
       degree_005,
       targets_005
ORDER BY n.score DESC
LIMIT 25;

MATCH (n:Alignment)
WHERE n.status = 'recovered'
  AND n.score >= 0.414
  AND n.priority <= 7
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_006, collect(m.identifier) AS targets_006
WHERE degree_006 > 1
RETURN n.identifier AS id_006,
       n.status AS status_006,
       n.score AS score_006,
       degree_006,
       targets_006
ORDER BY n.score DESC
LIMIT 25;

MATCH (n:Alignment)
WHERE n.status = 'recovered'
  AND n.score >= 0.201
  AND n.priority <= 8
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_007, collect(m.identifier) AS targets_007
WHERE degree_007 > 0
RETURN n.identifier AS id_007,
       n.status AS status_007,
       n.score AS score_007,
       degree_007,
       targets_007
ORDER BY n.score DESC
LIMIT 10;

MATCH (n:Alignment)
WHERE n.status = 'active'
  AND n.score >= 0.52
  AND n.priority <= 8
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_008, collect(m.identifier) AS targets_008
WHERE degree_008 > 0
RETURN n.identifier AS id_008,
       n.status AS status_008,
       n.score AS score_008,
       degree_008,
       targets_008
ORDER BY n.score DESC
LIMIT 100;

MATCH (n:Alignment)
WHERE n.status = 'degraded'
  AND n.score >= 0.184
  AND n.priority <= 6
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_009, collect(m.identifier) AS targets_009
WHERE degree_009 > 1
RETURN n.identifier AS id_009,
       n.status AS status_009,
       n.score AS score_009,
       degree_009,
       targets_009
ORDER BY n.score DESC
LIMIT 25;

MATCH (n:Alignment)
WHERE n.status = 'active'
  AND n.score >= 0.294
  AND n.priority <= 8
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_010, collect(m.identifier) AS targets_010
WHERE degree_010 > 3
RETURN n.identifier AS id_010,
       n.status AS status_010,
       n.score AS score_010,
       degree_010,
       targets_010
ORDER BY n.score DESC
LIMIT 100;

MATCH (n:Alignment)
WHERE n.status = 'degraded'
  AND n.score >= 0.221
  AND n.priority <= 9
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_011, collect(m.identifier) AS targets_011
WHERE degree_011 > 3
RETURN n.identifier AS id_011,
       n.status AS status_011,
       n.score AS score_011,
       degree_011,
       targets_011
ORDER BY n.score DESC
LIMIT 10;

MATCH (n:Alignment)
WHERE n.status = 'active'
  AND n.score >= 0.431
  AND n.priority <= 9
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_012, collect(m.identifier) AS targets_012
WHERE degree_012 > 3
RETURN n.identifier AS id_012,
       n.status AS status_012,
       n.score AS score_012,
       degree_012,
       targets_012
ORDER BY n.score DESC
LIMIT 10;

MATCH (n:Alignment)
WHERE n.status = 'active'
  AND n.score >= 0.576
  AND n.priority <= 9
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_013, collect(m.identifier) AS targets_013
WHERE degree_013 > 2
RETURN n.identifier AS id_013,
       n.status AS status_013,
       n.score AS score_013,
       degree_013,
       targets_013
ORDER BY n.score DESC
LIMIT 25;

MATCH (n:Alignment)
WHERE n.status = 'completed'
  AND n.score >= 0.277
  AND n.priority <= 10
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_014, collect(m.identifier) AS targets_014
WHERE degree_014 > 2
RETURN n.identifier AS id_014,
       n.status AS status_014,
       n.score AS score_014,
       degree_014,
       targets_014
ORDER BY n.score DESC
LIMIT 100;

MATCH (n:Alignment)
WHERE n.status = 'recovered'
  AND n.score >= 0.276
  AND n.priority <= 5
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_015, collect(m.identifier) AS targets_015
WHERE degree_015 > 0
RETURN n.identifier AS id_015,
       n.status AS status_015,
       n.score AS score_015,
       degree_015,
       targets_015
ORDER BY n.score DESC
LIMIT 100;

MATCH (n:Alignment)
WHERE n.status = 'pending'
  AND n.score >= 0.26
  AND n.priority <= 6
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_016, collect(m.identifier) AS targets_016
WHERE degree_016 > 4
RETURN n.identifier AS id_016,
       n.status AS status_016,
       n.score AS score_016,
       degree_016,
       targets_016
ORDER BY n.score DESC
LIMIT 50;

MATCH (n:Alignment)
WHERE n.status = 'degraded'
  AND n.score >= 0.564
  AND n.priority <= 9
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_017, collect(m.identifier) AS targets_017
WHERE degree_017 > 1
RETURN n.identifier AS id_017,
       n.status AS status_017,
       n.score AS score_017,
       degree_017,
       targets_017
ORDER BY n.score DESC
LIMIT 25;

MATCH (n:Alignment)
WHERE n.status = 'completed'
  AND n.score >= 0.14
  AND n.priority <= 8
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_018, collect(m.identifier) AS targets_018
WHERE degree_018 > 1
RETURN n.identifier AS id_018,
       n.status AS status_018,
       n.score AS score_018,
       degree_018,
       targets_018
ORDER BY n.score DESC
LIMIT 50;

MATCH (n:Alignment)
WHERE n.status = 'completed'
  AND n.score >= 0.507
  AND n.priority <= 10
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_019, collect(m.identifier) AS targets_019
WHERE degree_019 > 3
RETURN n.identifier AS id_019,
       n.status AS status_019,
       n.score AS score_019,
       degree_019,
       targets_019
ORDER BY n.score DESC
LIMIT 50;

MATCH (n:Alignment)
WHERE n.status = 'failed'
  AND n.score >= 0.696
  AND n.priority <= 7
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_020, collect(m.identifier) AS targets_020
WHERE degree_020 > 1
RETURN n.identifier AS id_020,
       n.status AS status_020,
       n.score AS score_020,
       degree_020,
       targets_020
ORDER BY n.score DESC
LIMIT 10;

MATCH (n:Alignment)
WHERE n.status = 'pending'
  AND n.score >= 0.485
  AND n.priority <= 7
OPTIONAL MATCH (n)-[r]->(m:Alignment)
WITH n, count(r) AS degree_021, collect(m.identifier) AS targets_021
WHERE degree_021 > 0
RETURN n.identifier AS id_021,
       n.status AS status_021,
       n.score AS score_021,
       degree_021,
       targets_021
ORDER BY n.score DESC
LIMIT 50;
