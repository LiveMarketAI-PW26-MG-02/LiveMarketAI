:param namespace => 'batchinference_02_02';
:param batchSize => 32;
:param threshold => 0.201;
:param maxDepth => 5;
:param timeoutSeconds => 84;
:param region => 'eu-west';
:param epoch => 22;
:param version => '3.6.9';

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_000' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.9536,
  latency: 73,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 964,
  confidence: 0.1176,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_001' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.8698,
  latency: 222,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 7734,
  confidence: 0.0102,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_002' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.2279,
  latency: 213,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 5583,
  confidence: 0.3876,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_003' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_004' })
MERGE (a)-[r_003:OBSERVES {
  strength: 0.5353,
  latency: 183,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 4200,
  confidence: 0.8566,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_004' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.2122,
  latency: 198,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8253,
  confidence: 0.0459,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_005' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.8481,
  latency: 130,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 116,
  confidence: 0.5237,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_006' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.807,
  latency: 233,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 4059,
  confidence: 0.7503,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_007' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.7192,
  latency: 19,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 1533,
  confidence: 0.9807,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_008' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.8705,
  latency: 125,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 410,
  confidence: 0.2035,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_009' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.0876,
  latency: 234,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 9153,
  confidence: 0.2194,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_010' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.6095,
  latency: 123,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 4808,
  confidence: 0.4472,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_011' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_012' })
MERGE (a)-[r_011:VALIDATES {
  strength: 0.7555,
  latency: 131,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 2152,
  confidence: 0.2855,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_012' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.1054,
  latency: 103,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 2673,
  confidence: 0.6514,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_013' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.8521,
  latency: 205,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 5729,
  confidence: 0.6146,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_014' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.8455,
  latency: 203,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 4528,
  confidence: 0.4019,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_015' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.2494,
  latency: 141,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 2614,
  confidence: 0.9016,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_016' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.9553,
  latency: 224,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 804,
  confidence: 0.2375,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_017' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.8381,
  latency: 183,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 3906,
  confidence: 0.7133,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_018' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.7159,
  latency: 153,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9912,
  confidence: 0.8177,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_019' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.4576,
  latency: 226,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 9344,
  confidence: 0.7513,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_020' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.1188,
  latency: 13,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 6560,
  confidence: 0.1286,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_021' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.5088,
  latency: 34,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 2544,
  confidence: 0.1628,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_022' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.8806,
  latency: 19,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 9920,
  confidence: 0.9647,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_023' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.2898,
  latency: 130,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 3246,
  confidence: 0.9234,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_024' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.107,
  latency: 228,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 772,
  confidence: 0.5513,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_025' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.9852,
  latency: 146,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 3912,
  confidence: 0.6679,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_026' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.5768,
  latency: 48,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6963,
  confidence: 0.9533,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_027' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.9984,
  latency: 201,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 3971,
  confidence: 0.4316,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_028' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.6004,
  latency: 20,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 5137,
  confidence: 0.1644,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_01_core_engine_2_029' }),
      (b:BatchInference { identifier: 'batchinference_01_core_engine_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.4375,
  latency: 250,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 2481,
  confidence: 0.7413,
  active: true
}]->(b);
