:param namespace => 'batchinference_02_02';
:param batchSize => 64;
:param threshold => 0.658;
:param maxDepth => 5;
:param timeoutSeconds => 107;
:param region => 'us-west';
:param epoch => 11;
:param version => '5.7.6';

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_000' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.8012,
  latency: 155,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 9755,
  confidence: 0.1172,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_001' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.688,
  latency: 61,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 1185,
  confidence: 0.9556,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_002' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.4287,
  latency: 98,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 9119,
  confidence: 0.6794,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_003' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.1475,
  latency: 67,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 5020,
  confidence: 0.7228,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_004' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.1402,
  latency: 185,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 1896,
  confidence: 0.1808,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_005' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.7402,
  latency: 161,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 6776,
  confidence: 0.7885,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_006' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.6978,
  latency: 171,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 1279,
  confidence: 0.4239,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_007' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.2199,
  latency: 99,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4763,
  confidence: 0.8476,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_008' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.7401,
  latency: 230,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 5405,
  confidence: 0.0382,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_009' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.1274,
  latency: 167,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 9221,
  confidence: 0.2299,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_010' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.8965,
  latency: 205,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 5762,
  confidence: 0.4398,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_011' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.7006,
  latency: 222,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 9773,
  confidence: 0.1095,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_012' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.4767,
  latency: 21,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 3087,
  confidence: 0.3851,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_013' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.8159,
  latency: 47,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 8068,
  confidence: 0.015,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_014' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.1701,
  latency: 92,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 2082,
  confidence: 0.4,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_015' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.4536,
  latency: 90,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 756,
  confidence: 0.07,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_016' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.2164,
  latency: 92,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 6571,
  confidence: 0.1888,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_017' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.0896,
  latency: 149,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 1099,
  confidence: 0.618,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_018' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.1793,
  latency: 80,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 4244,
  confidence: 0.9254,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_019' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.6717,
  latency: 249,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 7786,
  confidence: 0.0636,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_020' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.1252,
  latency: 21,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 6248,
  confidence: 0.6206,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_021' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.3107,
  latency: 166,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 8387,
  confidence: 0.8396,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_022' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.5963,
  latency: 163,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 4283,
  confidence: 0.64,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_023' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.8499,
  latency: 223,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 3203,
  confidence: 0.6405,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_024' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.1868,
  latency: 144,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 6945,
  confidence: 0.2663,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_025' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.8056,
  latency: 245,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 1245,
  confidence: 0.7823,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_026' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.0304,
  latency: 93,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 9303,
  confidence: 0.8015,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_027' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.6334,
  latency: 194,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 2993,
  confidence: 0.8557,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_028' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.7981,
  latency: 51,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 8492,
  confidence: 0.577,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_02_state_handlers_2_029' }),
      (b:BatchInference { identifier: 'batchinference_02_state_handlers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.9803,
  latency: 135,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 9520,
  confidence: 0.4323,
  active: true
}]->(b);
