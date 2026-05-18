:param namespace => 'transformer_02_02';
:param batchSize => 512;
:param threshold => 0.587;
:param maxDepth => 12;
:param timeoutSeconds => 15;
:param region => 'us-west';
:param epoch => 25;
:param version => '3.7.5';

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_000' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.1387,
  latency: 69,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5608,
  confidence: 0.5991,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_001' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.0189,
  latency: 61,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 5175,
  confidence: 0.6789,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_002' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.1022,
  latency: 173,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 4138,
  confidence: 0.806,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_003' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.7609,
  latency: 108,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 3072,
  confidence: 0.1219,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_004' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.4194,
  latency: 218,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 9654,
  confidence: 0.1304,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_005' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.134,
  latency: 185,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 503,
  confidence: 0.9613,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_006' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.2067,
  latency: 66,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 5066,
  confidence: 0.6282,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_007' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.5296,
  latency: 81,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 917,
  confidence: 0.8839,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_008' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.1396,
  latency: 192,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 7169,
  confidence: 0.0242,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_009' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.6696,
  latency: 246,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 9596,
  confidence: 0.9351,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_010' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.9744,
  latency: 206,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 5761,
  confidence: 0.1035,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_011' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.0637,
  latency: 237,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 8489,
  confidence: 0.6073,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_012' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.5959,
  latency: 57,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 1391,
  confidence: 0.566,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_013' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.4514,
  latency: 32,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 9259,
  confidence: 0.1124,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_014' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.2623,
  latency: 225,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 9106,
  confidence: 0.4367,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_015' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.7105,
  latency: 59,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 5605,
  confidence: 0.7786,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_016' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.298,
  latency: 183,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 3399,
  confidence: 0.0079,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_017' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.7763,
  latency: 85,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 1124,
  confidence: 0.719,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_018' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.6491,
  latency: 185,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 8121,
  confidence: 0.9118,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_019' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.9785,
  latency: 167,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 8759,
  confidence: 0.1511,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_020' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.1017,
  latency: 195,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 1623,
  confidence: 0.3965,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_021' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.0173,
  latency: 34,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 4194,
  confidence: 0.5545,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_022' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.2276,
  latency: 135,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 158,
  confidence: 0.4873,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_023' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.9624,
  latency: 227,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6653,
  confidence: 0.6558,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_024' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.5384,
  latency: 216,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 7188,
  confidence: 0.1164,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_025' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.3196,
  latency: 236,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 6503,
  confidence: 0.0551,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_026' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.7826,
  latency: 15,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 8934,
  confidence: 0.7275,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_027' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.718,
  latency: 88,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 5301,
  confidence: 0.3816,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_028' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.3694,
  latency: 135,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 6352,
  confidence: 0.8386,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_02_state_handlers_2_029' }),
      (b:Transformer { identifier: 'transformer_02_state_handlers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.7522,
  latency: 101,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 7819,
  confidence: 0.1545,
  active: true
}]->(b);
