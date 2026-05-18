:param namespace => 'transformer_02_02';
:param batchSize => 256;
:param threshold => 0.579;
:param maxDepth => 8;
:param timeoutSeconds => 64;
:param region => 'us-west';
:param epoch => 65;
:param version => '4.9.1';

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_000' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.121,
  latency: 70,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 1499,
  confidence: 0.5011,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_001' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.8934,
  latency: 204,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 8274,
  confidence: 0.6264,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_002' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.5202,
  latency: 231,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 128,
  confidence: 0.0469,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_003' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.7635,
  latency: 175,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 8349,
  confidence: 0.004,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_004' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.9503,
  latency: 67,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 5732,
  confidence: 0.6787,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_005' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.0433,
  latency: 207,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 3837,
  confidence: 0.7775,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_006' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.2732,
  latency: 234,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 353,
  confidence: 0.4832,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_007' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.4477,
  latency: 22,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6440,
  confidence: 0.1879,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_008' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.2397,
  latency: 164,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 6930,
  confidence: 0.5592,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_009' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.5349,
  latency: 28,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 2618,
  confidence: 0.9248,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_010' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.9811,
  latency: 15,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 8085,
  confidence: 0.0316,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_011' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_012' })
MERGE (a)-[r_011:DEPENDS_ON {
  strength: 0.955,
  latency: 119,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 4559,
  confidence: 0.3152,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_012' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.7971,
  latency: 36,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 8782,
  confidence: 0.4682,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_013' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.9302,
  latency: 94,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 163,
  confidence: 0.3058,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_014' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.8462,
  latency: 152,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 4316,
  confidence: 0.2029,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_015' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.9774,
  latency: 87,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 8479,
  confidence: 0.8738,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_016' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.6135,
  latency: 155,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 2221,
  confidence: 0.5043,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_017' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.2455,
  latency: 194,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 9000,
  confidence: 0.8267,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_018' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.5545,
  latency: 2,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 3977,
  confidence: 0.7923,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_019' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.4844,
  latency: 63,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 9256,
  confidence: 0.1354,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_020' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.8423,
  latency: 176,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 7306,
  confidence: 0.0551,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_021' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.2239,
  latency: 181,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 3921,
  confidence: 0.8604,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_022' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.8139,
  latency: 79,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 5700,
  confidence: 0.1812,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_023' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.8327,
  latency: 117,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 9113,
  confidence: 0.8286,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_024' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.1241,
  latency: 87,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 4466,
  confidence: 0.1791,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_025' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.0796,
  latency: 135,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 663,
  confidence: 0.1604,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_026' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.4399,
  latency: 96,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 5150,
  confidence: 0.3057,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_027' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.2606,
  latency: 33,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 7631,
  confidence: 0.4165,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_028' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.7395,
  latency: 73,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 6818,
  confidence: 0.0333,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_10_utility_helpers_2_029' }),
      (b:Transformer { identifier: 'transformer_10_utility_helpers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.4194,
  latency: 30,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 5551,
  confidence: 0.18,
  active: true
}]->(b);
