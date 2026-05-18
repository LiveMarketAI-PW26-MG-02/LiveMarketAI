:param namespace => 'inferencecontext_02_02';
:param batchSize => 256;
:param threshold => 0.337;
:param maxDepth => 4;
:param timeoutSeconds => 71;
:param region => 'eu-west';
:param epoch => 86;
:param version => '5.4.5';

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_000' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_001' })
MERGE (a)-[r_000:PRODUCES {
  strength: 0.4069,
  latency: 99,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 9958,
  confidence: 0.6441,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_001' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.5387,
  latency: 228,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 3022,
  confidence: 0.6813,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_002' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.5788,
  latency: 100,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 6390,
  confidence: 0.2542,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_003' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_004' })
MERGE (a)-[r_003:PRODUCES {
  strength: 0.8813,
  latency: 200,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 4215,
  confidence: 0.4942,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_004' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.3291,
  latency: 12,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 6916,
  confidence: 0.133,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_005' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.0761,
  latency: 83,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 8917,
  confidence: 0.7374,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_006' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.3452,
  latency: 168,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 6812,
  confidence: 0.7352,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_007' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.5595,
  latency: 15,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 5299,
  confidence: 0.156,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_008' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.6907,
  latency: 66,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 7222,
  confidence: 0.3809,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_009' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.6257,
  latency: 27,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 1185,
  confidence: 0.3645,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_010' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.2115,
  latency: 3,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 7249,
  confidence: 0.9469,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_011' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.1135,
  latency: 199,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 6105,
  confidence: 0.9739,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_012' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.6713,
  latency: 1,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 2784,
  confidence: 0.1211,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_013' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.9503,
  latency: 228,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 7031,
  confidence: 0.6401,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_014' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.5255,
  latency: 105,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 9017,
  confidence: 0.3392,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_015' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.2717,
  latency: 173,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 2768,
  confidence: 0.3884,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_016' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.2366,
  latency: 156,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 2155,
  confidence: 0.1642,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_017' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.8686,
  latency: 10,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 949,
  confidence: 0.8774,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_018' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.6265,
  latency: 71,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 6171,
  confidence: 0.589,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_019' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.9432,
  latency: 113,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 7490,
  confidence: 0.8544,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_020' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.2855,
  latency: 93,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 9504,
  confidence: 0.9822,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_021' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.6987,
  latency: 198,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 6610,
  confidence: 0.642,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_022' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.0466,
  latency: 6,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 7140,
  confidence: 0.8141,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_023' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.6848,
  latency: 181,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 3770,
  confidence: 0.524,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_024' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.2258,
  latency: 100,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 4668,
  confidence: 0.3888,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_025' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.9474,
  latency: 204,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 3397,
  confidence: 0.0683,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_026' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.8185,
  latency: 228,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 713,
  confidence: 0.9888,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_027' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.9274,
  latency: 64,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 6507,
  confidence: 0.4066,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_028' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.7216,
  latency: 19,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 1213,
  confidence: 0.0948,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_029' }),
      (b:InferenceContext { identifier: 'inferencecontext_01_core_engine_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.4925,
  latency: 195,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 5532,
  confidence: 0.5806,
  active: true
}]->(b);
