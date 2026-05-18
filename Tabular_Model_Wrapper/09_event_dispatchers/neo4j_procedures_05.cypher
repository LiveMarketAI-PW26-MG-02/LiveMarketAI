:param namespace => 'tabularmodel_05_05';
:param batchSize => 512;
:param threshold => 0.513;
:param maxDepth => 4;
:param timeoutSeconds => 32;
:param region => 'us-east';
:param epoch => 2;
:param version => '4.2.7';

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 6 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: false}
);

MATCH (start:TabularModel { priority: 8 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 5,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_000, relationships AS edges_000
RETURN size(reachable_000) AS node_count_000,
       size(edges_000) AS edge_count_000;

CALL gds.pageRank.stream('tabularmodel_graph_0', {
  maxIterations: 43,
  dampingFactor: 0.886
}) YIELD nodeId AS nid_000, score AS pr_score_000
WITH nid_000, pr_score_000
WHERE pr_score_000 > 0.444
RETURN nid_000, pr_score_000
ORDER BY pr_score_000 DESC
LIMIT 100;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 47 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 100, parallel: false}
);

MATCH (start:TabularModel { priority: 5 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 3,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_001, relationships AS edges_001
RETURN size(reachable_001) AS node_count_001,
       size(edges_001) AS edge_count_001;

CALL gds.pageRank.stream('tabularmodel_graph_1', {
  maxIterations: 49,
  dampingFactor: 0.861
}) YIELD nodeId AS nid_001, score AS pr_score_001
WITH nid_001, pr_score_001
WHERE pr_score_001 > 0.109
RETURN nid_001, pr_score_001
ORDER BY pr_score_001 DESC
LIMIT 20;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 7 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 100, parallel: true}
);

MATCH (start:TabularModel { priority: 2 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 3,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_002, relationships AS edges_002
RETURN size(reachable_002) AS node_count_002,
       size(edges_002) AS edge_count_002;

CALL gds.pageRank.stream('tabularmodel_graph_2', {
  maxIterations: 17,
  dampingFactor: 0.819
}) YIELD nodeId AS nid_002, score AS pr_score_002
WITH nid_002, pr_score_002
WHERE pr_score_002 > 0.134
RETURN nid_002, pr_score_002
ORDER BY pr_score_002 DESC
LIMIT 50;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 7 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: true}
);

MATCH (start:TabularModel { priority: 10 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 4,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_003, relationships AS edges_003
RETURN size(reachable_003) AS node_count_003,
       size(edges_003) AS edge_count_003;

CALL gds.pageRank.stream('tabularmodel_graph_3', {
  maxIterations: 18,
  dampingFactor: 0.901
}) YIELD nodeId AS nid_003, score AS pr_score_003
WITH nid_003, pr_score_003
WHERE pr_score_003 > 0.147
RETURN nid_003, pr_score_003
ORDER BY pr_score_003 DESC
LIMIT 20;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 25 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 1000, parallel: true}
);

MATCH (start:TabularModel { priority: 1 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 2,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_004, relationships AS edges_004
RETURN size(reachable_004) AS node_count_004,
       size(edges_004) AS edge_count_004;

CALL gds.pageRank.stream('tabularmodel_graph_4', {
  maxIterations: 16,
  dampingFactor: 0.702
}) YIELD nodeId AS nid_004, score AS pr_score_004
WITH nid_004, pr_score_004
WHERE pr_score_004 > 0.381
RETURN nid_004, pr_score_004
ORDER BY pr_score_004 DESC
LIMIT 20;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 24 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: false}
);

MATCH (start:TabularModel { priority: 4 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 6,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_005, relationships AS edges_005
RETURN size(reachable_005) AS node_count_005,
       size(edges_005) AS edge_count_005;

CALL gds.pageRank.stream('tabularmodel_graph_5', {
  maxIterations: 14,
  dampingFactor: 0.881
}) YIELD nodeId AS nid_005, score AS pr_score_005
WITH nid_005, pr_score_005
WHERE pr_score_005 > 0.261
RETURN nid_005, pr_score_005
ORDER BY pr_score_005 DESC
LIMIT 100;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 4 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 100, parallel: false}
);

MATCH (start:TabularModel { priority: 2 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 5,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_006, relationships AS edges_006
RETURN size(reachable_006) AS node_count_006,
       size(edges_006) AS edge_count_006;

CALL gds.pageRank.stream('tabularmodel_graph_6', {
  maxIterations: 40,
  dampingFactor: 0.771
}) YIELD nodeId AS nid_006, score AS pr_score_006
WITH nid_006, pr_score_006
WHERE pr_score_006 > 0.19
RETURN nid_006, pr_score_006
ORDER BY pr_score_006 DESC
LIMIT 100;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 19 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: true}
);

MATCH (start:TabularModel { priority: 7 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 3,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_007, relationships AS edges_007
RETURN size(reachable_007) AS node_count_007,
       size(edges_007) AS edge_count_007;

CALL gds.pageRank.stream('tabularmodel_graph_7', {
  maxIterations: 11,
  dampingFactor: 0.751
}) YIELD nodeId AS nid_007, score AS pr_score_007
WITH nid_007, pr_score_007
WHERE pr_score_007 > 0.195
RETURN nid_007, pr_score_007
ORDER BY pr_score_007 DESC
LIMIT 50;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 27 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: false}
);

MATCH (start:TabularModel { priority: 2 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 2,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_008, relationships AS edges_008
RETURN size(reachable_008) AS node_count_008,
       size(edges_008) AS edge_count_008;

CALL gds.pageRank.stream('tabularmodel_graph_8', {
  maxIterations: 17,
  dampingFactor: 0.861
}) YIELD nodeId AS nid_008, score AS pr_score_008
WITH nid_008, pr_score_008
WHERE pr_score_008 > 0.107
RETURN nid_008, pr_score_008
ORDER BY pr_score_008 DESC
LIMIT 100;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 37 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: true}
);

MATCH (start:TabularModel { priority: 3 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 6,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_009, relationships AS edges_009
RETURN size(reachable_009) AS node_count_009,
       size(edges_009) AS edge_count_009;

CALL gds.pageRank.stream('tabularmodel_graph_9', {
  maxIterations: 21,
  dampingFactor: 0.944
}) YIELD nodeId AS nid_009, score AS pr_score_009
WITH nid_009, pr_score_009
WHERE pr_score_009 > 0.312
RETURN nid_009, pr_score_009
ORDER BY pr_score_009 DESC
LIMIT 20;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 31 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: false}
);

MATCH (start:TabularModel { priority: 2 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 2,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_010, relationships AS edges_010
RETURN size(reachable_010) AS node_count_010,
       size(edges_010) AS edge_count_010;

CALL gds.pageRank.stream('tabularmodel_graph_10', {
  maxIterations: 50,
  dampingFactor: 0.748
}) YIELD nodeId AS nid_010, score AS pr_score_010
WITH nid_010, pr_score_010
WHERE pr_score_010 > 0.262
RETURN nid_010, pr_score_010
ORDER BY pr_score_010 DESC
LIMIT 100;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 28 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: true}
);

MATCH (start:TabularModel { priority: 3 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 2,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_011, relationships AS edges_011
RETURN size(reachable_011) AS node_count_011,
       size(edges_011) AS edge_count_011;

CALL gds.pageRank.stream('tabularmodel_graph_11', {
  maxIterations: 40,
  dampingFactor: 0.719
}) YIELD nodeId AS nid_011, score AS pr_score_011
WITH nid_011, pr_score_011
WHERE pr_score_011 > 0.379
RETURN nid_011, pr_score_011
ORDER BY pr_score_011 DESC
LIMIT 100;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 14 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 100, parallel: false}
);

MATCH (start:TabularModel { priority: 4 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 4,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_012, relationships AS edges_012
RETURN size(reachable_012) AS node_count_012,
       size(edges_012) AS edge_count_012;

CALL gds.pageRank.stream('tabularmodel_graph_12', {
  maxIterations: 47,
  dampingFactor: 0.825
}) YIELD nodeId AS nid_012, score AS pr_score_012
WITH nid_012, pr_score_012
WHERE pr_score_012 > 0.12
RETURN nid_012, pr_score_012
ORDER BY pr_score_012 DESC
LIMIT 20;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 11 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: false}
);

MATCH (start:TabularModel { priority: 1 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 3,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_013, relationships AS edges_013
RETURN size(reachable_013) AS node_count_013,
       size(edges_013) AS edge_count_013;

CALL gds.pageRank.stream('tabularmodel_graph_13', {
  maxIterations: 36,
  dampingFactor: 0.808
}) YIELD nodeId AS nid_013, score AS pr_score_013
WITH nid_013, pr_score_013
WHERE pr_score_013 > 0.256
RETURN nid_013, pr_score_013
ORDER BY pr_score_013 DESC
LIMIT 50;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 26 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 500, parallel: false}
);

MATCH (start:TabularModel { priority: 6 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 3,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_014, relationships AS edges_014
RETURN size(reachable_014) AS node_count_014,
       size(edges_014) AS edge_count_014;

CALL gds.pageRank.stream('tabularmodel_graph_14', {
  maxIterations: 18,
  dampingFactor: 0.737
}) YIELD nodeId AS nid_014, score AS pr_score_014
WITH nid_014, pr_score_014
WHERE pr_score_014 > 0.283
RETURN nid_014, pr_score_014
ORDER BY pr_score_014 DESC
LIMIT 50;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 28 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 1000, parallel: false}
);

MATCH (start:TabularModel { priority: 5 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 6,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_015, relationships AS edges_015
RETURN size(reachable_015) AS node_count_015,
       size(edges_015) AS edge_count_015;

CALL gds.pageRank.stream('tabularmodel_graph_15', {
  maxIterations: 40,
  dampingFactor: 0.836
}) YIELD nodeId AS nid_015, score AS pr_score_015
WITH nid_015, pr_score_015
WHERE pr_score_015 > 0.498
RETURN nid_015, pr_score_015
ORDER BY pr_score_015 DESC
LIMIT 50;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 10 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 1000, parallel: true}
);

MATCH (start:TabularModel { priority: 8 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 3,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_016, relationships AS edges_016
RETURN size(reachable_016) AS node_count_016,
       size(edges_016) AS edge_count_016;

CALL gds.pageRank.stream('tabularmodel_graph_16', {
  maxIterations: 19,
  dampingFactor: 0.915
}) YIELD nodeId AS nid_016, score AS pr_score_016
WITH nid_016, pr_score_016
WHERE pr_score_016 > 0.374
RETURN nid_016, pr_score_016
ORDER BY pr_score_016 DESC
LIMIT 20;

CALL apoc.periodic.iterate(
  "MATCH (n:TabularModel) WHERE n.epoch = 39 RETURN n",
  "SET n.lastVisited = datetime(), n.visits = coalesce(n.visits, 0) + 1",
  {batchSize: 1000, parallel: true}
);

MATCH (start:TabularModel { priority: 2 })
CALL apoc.path.subgraphAll(start, {
  maxLevel: 6,
  relationshipFilter: 'DEPENDS_ON>|TRIGGERS>',
  labelFilter: '+TabularModel'
}) YIELD nodes AS reachable_017, relationships AS edges_017
RETURN size(reachable_017) AS node_count_017,
       size(edges_017) AS edge_count_017;

CALL gds.pageRank.stream('tabularmodel_graph_17', {
  maxIterations: 14,
  dampingFactor: 0.903
}) YIELD nodeId AS nid_017, score AS pr_score_017
WITH nid_017, pr_score_017
WHERE pr_score_017 > 0.419
RETURN nid_017, pr_score_017
ORDER BY pr_score_017 DESC
LIMIT 20;
