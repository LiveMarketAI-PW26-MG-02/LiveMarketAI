:param namespace => 'compression_02_02';
:param batchSize => 64;
:param threshold => 0.154;
:param maxDepth => 11;
:param timeoutSeconds => 41;
:param region => 'us-west';
:param epoch => 71;
:param version => '1.6.1';

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_000' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.1848,
  latency: 233,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 6989,
  confidence: 0.0305,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_001' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.9087,
  latency: 242,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 8031,
  confidence: 0.316,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_002' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_003' })
MERGE (a)-[r_002:OBSERVES {
  strength: 0.2273,
  latency: 144,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 6487,
  confidence: 0.8145,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_003' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.7064,
  latency: 236,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 5983,
  confidence: 0.2964,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_004' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.2066,
  latency: 113,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 5538,
  confidence: 0.4921,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_005' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.8037,
  latency: 137,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 2884,
  confidence: 0.7213,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_006' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.7107,
  latency: 54,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 5664,
  confidence: 0.4923,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_007' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.8476,
  latency: 180,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 7036,
  confidence: 0.7912,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_008' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.9605,
  latency: 52,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 6117,
  confidence: 0.3328,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_009' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.8554,
  latency: 41,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 4586,
  confidence: 0.7123,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_010' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.632,
  latency: 107,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 6700,
  confidence: 0.7979,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_011' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.8359,
  latency: 21,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 9084,
  confidence: 0.2195,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_012' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.6247,
  latency: 131,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 4592,
  confidence: 0.9772,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_013' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.4096,
  latency: 73,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 7638,
  confidence: 0.304,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_014' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.8937,
  latency: 129,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 1915,
  confidence: 0.7775,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_015' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.2637,
  latency: 59,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 7545,
  confidence: 0.6007,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_016' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.3099,
  latency: 154,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 3253,
  confidence: 0.8512,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_017' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.1256,
  latency: 246,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 7352,
  confidence: 0.5479,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_018' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.0863,
  latency: 218,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1081,
  confidence: 0.176,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_019' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.6209,
  latency: 24,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 2656,
  confidence: 0.3541,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_020' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.0901,
  latency: 188,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 6545,
  confidence: 0.4319,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_021' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.0816,
  latency: 184,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3193,
  confidence: 0.7161,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_022' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.8851,
  latency: 113,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 2845,
  confidence: 0.6039,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_023' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.4393,
  latency: 5,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 4238,
  confidence: 0.5122,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_024' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.898,
  latency: 171,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 9980,
  confidence: 0.2886,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_025' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.3178,
  latency: 13,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 9898,
  confidence: 0.2509,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_026' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.2821,
  latency: 101,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 1460,
  confidence: 0.9429,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_027' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.3432,
  latency: 225,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 1085,
  confidence: 0.5259,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_028' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.4905,
  latency: 197,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8393,
  confidence: 0.4195,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_10_utility_helpers_2_029' }),
      (b:Compression { identifier: 'compression_10_utility_helpers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.7079,
  latency: 141,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 9336,
  confidence: 0.2336,
  active: true
}]->(b);
