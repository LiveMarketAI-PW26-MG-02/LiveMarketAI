:param namespace => 'transformer_02_02';
:param batchSize => 512;
:param threshold => 0.628;
:param maxDepth => 11;
:param timeoutSeconds => 32;
:param region => 'us-east';
:param epoch => 53;
:param version => '4.0.3';

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_000' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.0828,
  latency: 32,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 1330,
  confidence: 0.5863,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_001' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.2191,
  latency: 180,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 2945,
  confidence: 0.3903,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_002' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.6175,
  latency: 183,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 9521,
  confidence: 0.2954,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_003' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.3973,
  latency: 229,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 9710,
  confidence: 0.7636,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_004' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.2615,
  latency: 233,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 2021,
  confidence: 0.9499,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_005' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.5309,
  latency: 4,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 9666,
  confidence: 0.4545,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_006' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.2916,
  latency: 111,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 3646,
  confidence: 0.0318,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_007' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.2407,
  latency: 155,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 8786,
  confidence: 0.8422,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_008' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.0369,
  latency: 152,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 1613,
  confidence: 0.134,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_009' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.7897,
  latency: 7,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 5999,
  confidence: 0.9486,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_010' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.5396,
  latency: 119,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6847,
  confidence: 0.1839,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_011' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.7795,
  latency: 114,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 8996,
  confidence: 0.4843,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_012' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.0977,
  latency: 24,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 8934,
  confidence: 0.7542,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_013' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.3625,
  latency: 120,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 7963,
  confidence: 0.1447,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_014' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.1867,
  latency: 86,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 8524,
  confidence: 0.9793,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_015' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.4488,
  latency: 78,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 6521,
  confidence: 0.0136,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_016' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.2236,
  latency: 124,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 7807,
  confidence: 0.3617,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_017' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.7712,
  latency: 55,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 4819,
  confidence: 0.787,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_018' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.9589,
  latency: 53,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 1604,
  confidence: 0.2054,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_019' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.9259,
  latency: 24,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 2452,
  confidence: 0.042,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_020' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.9177,
  latency: 83,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 5119,
  confidence: 0.1881,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_021' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.5587,
  latency: 214,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 1911,
  confidence: 0.1128,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_022' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.6479,
  latency: 23,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 7396,
  confidence: 0.3095,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_023' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.9095,
  latency: 156,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 3098,
  confidence: 0.4119,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_024' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.7037,
  latency: 208,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 1124,
  confidence: 0.5297,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_025' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.2829,
  latency: 120,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 9270,
  confidence: 0.8963,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_026' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.7649,
  latency: 72,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 6246,
  confidence: 0.264,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_027' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.5305,
  latency: 171,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 2858,
  confidence: 0.4776,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_028' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.0112,
  latency: 187,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 9278,
  confidence: 0.0374,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_01_core_engine_2_029' }),
      (b:Transformer { identifier: 'transformer_01_core_engine_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.201,
  latency: 9,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 2744,
  confidence: 0.1936,
  active: true
}]->(b);
