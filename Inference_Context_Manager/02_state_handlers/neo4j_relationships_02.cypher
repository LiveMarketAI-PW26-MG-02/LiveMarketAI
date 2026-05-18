:param namespace => 'inferencecontext_02_02';
:param batchSize => 32;
:param threshold => 0.149;
:param maxDepth => 12;
:param timeoutSeconds => 51;
:param region => 'us-west';
:param epoch => 54;
:param version => '4.6.0';

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_000' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.3421,
  latency: 203,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 6163,
  confidence: 0.1747,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_001' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.9633,
  latency: 52,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 7042,
  confidence: 0.6872,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_002' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.9547,
  latency: 170,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 3140,
  confidence: 0.5959,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_003' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.9924,
  latency: 98,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 6656,
  confidence: 0.6595,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_004' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.8861,
  latency: 124,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 7225,
  confidence: 0.2986,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_005' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.6574,
  latency: 41,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 3708,
  confidence: 0.8227,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_006' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.3532,
  latency: 110,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 5250,
  confidence: 0.7946,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_007' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.5341,
  latency: 199,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 1888,
  confidence: 0.9919,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_008' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.5891,
  latency: 194,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 8492,
  confidence: 0.7844,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_009' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.4035,
  latency: 223,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6671,
  confidence: 0.0146,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_010' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.9655,
  latency: 151,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 5775,
  confidence: 0.9469,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_011' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_012' })
MERGE (a)-[r_011:DEPENDS_ON {
  strength: 0.8016,
  latency: 141,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 7106,
  confidence: 0.1821,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_012' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.6393,
  latency: 60,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 4572,
  confidence: 0.7949,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_013' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.0774,
  latency: 215,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 5020,
  confidence: 0.5951,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_014' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.8246,
  latency: 161,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 2081,
  confidence: 0.2067,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_015' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.648,
  latency: 203,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 6388,
  confidence: 0.8887,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_016' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.2571,
  latency: 132,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 144,
  confidence: 0.183,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_017' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.3077,
  latency: 136,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 9836,
  confidence: 0.0592,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_018' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.1721,
  latency: 195,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 3732,
  confidence: 0.7685,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_019' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_020' })
MERGE (a)-[r_019:PRODUCES {
  strength: 0.3299,
  latency: 212,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 4186,
  confidence: 0.8982,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_020' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.0844,
  latency: 58,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 4002,
  confidence: 0.9664,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_021' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.6459,
  latency: 38,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 5373,
  confidence: 0.2713,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_022' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.3776,
  latency: 119,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 5123,
  confidence: 0.8973,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_023' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.6237,
  latency: 55,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 743,
  confidence: 0.4557,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_024' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.7949,
  latency: 96,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 556,
  confidence: 0.867,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_025' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.416,
  latency: 84,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 3313,
  confidence: 0.6256,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_026' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.0885,
  latency: 185,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 1311,
  confidence: 0.7086,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_027' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.5822,
  latency: 231,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 5007,
  confidence: 0.7378,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_028' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.5213,
  latency: 17,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 7978,
  confidence: 0.158,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_029' }),
      (b:InferenceContext { identifier: 'inferencecontext_02_state_handlers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.1577,
  latency: 115,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 7204,
  confidence: 0.4816,
  active: true
}]->(b);
