:param namespace => 'transformer_02_02';
:param batchSize => 64;
:param threshold => 0.333;
:param maxDepth => 4;
:param timeoutSeconds => 31;
:param region => 'eu-west';
:param epoch => 57;
:param version => '2.2.4';

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_000' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:PRODUCES {
  strength: 0.2094,
  latency: 114,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 9996,
  confidence: 0.5316,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_001' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.3754,
  latency: 167,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 4007,
  confidence: 0.0828,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_002' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.7466,
  latency: 10,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 5200,
  confidence: 0.4001,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_003' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.9412,
  latency: 104,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 6372,
  confidence: 0.9831,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_004' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.5884,
  latency: 31,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 2842,
  confidence: 0.13,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_005' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.9969,
  latency: 99,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 2482,
  confidence: 0.5829,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_006' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.4722,
  latency: 204,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 117,
  confidence: 0.0378,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_007' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.0323,
  latency: 164,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 1292,
  confidence: 0.3368,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_008' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.4327,
  latency: 35,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 7716,
  confidence: 0.2475,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_009' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.669,
  latency: 129,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 249,
  confidence: 0.3545,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_010' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.3376,
  latency: 91,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 4381,
  confidence: 0.7767,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_011' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.6431,
  latency: 42,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 6108,
  confidence: 0.0858,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_012' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.6095,
  latency: 247,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 8846,
  confidence: 0.372,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_013' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.9938,
  latency: 188,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3955,
  confidence: 0.5531,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_014' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.0705,
  latency: 19,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 1665,
  confidence: 0.1474,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_015' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.961,
  latency: 72,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 3785,
  confidence: 0.1606,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_016' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.8094,
  latency: 79,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 5734,
  confidence: 0.4429,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_017' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.2797,
  latency: 7,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 5225,
  confidence: 0.5261,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_018' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.4091,
  latency: 243,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 819,
  confidence: 0.5136,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_019' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.6999,
  latency: 222,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 929,
  confidence: 0.7194,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_020' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.0844,
  latency: 30,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 6613,
  confidence: 0.9285,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_021' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.7472,
  latency: 22,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 2031,
  confidence: 0.0059,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_022' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.1298,
  latency: 78,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 8940,
  confidence: 0.4071,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_023' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.2437,
  latency: 200,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4902,
  confidence: 0.0908,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_024' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.6362,
  latency: 210,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 4118,
  confidence: 0.7611,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_025' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.9943,
  latency: 235,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 5374,
  confidence: 0.2133,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_026' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.0863,
  latency: 59,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1808,
  confidence: 0.0098,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_027' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.3844,
  latency: 72,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 8306,
  confidence: 0.3153,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_028' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.5585,
  latency: 221,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 2475,
  confidence: 0.7085,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_09_event_dispatchers_2_029' }),
      (b:Transformer { identifier: 'transformer_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:VALIDATES {
  strength: 0.5093,
  latency: 141,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 4983,
  confidence: 0.2614,
  active: true
}]->(b);
