:param namespace => 'compression_02_02';
:param batchSize => 128;
:param threshold => 0.689;
:param maxDepth => 12;
:param timeoutSeconds => 102;
:param region => 'eu-west';
:param epoch => 39;
:param version => '2.4.1';

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_000' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.702,
  latency: 204,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 4542,
  confidence: 0.7319,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_001' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.5622,
  latency: 22,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 5049,
  confidence: 0.4399,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_002' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.5392,
  latency: 217,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 1277,
  confidence: 0.1054,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_003' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.1973,
  latency: 58,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 1482,
  confidence: 0.1764,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_004' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.0906,
  latency: 174,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 2702,
  confidence: 0.923,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_005' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.2086,
  latency: 21,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 397,
  confidence: 0.0221,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_006' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.6042,
  latency: 123,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 150,
  confidence: 0.2557,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_007' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.0629,
  latency: 78,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 489,
  confidence: 0.6175,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_008' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.7768,
  latency: 75,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 9621,
  confidence: 0.8241,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_009' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.5714,
  latency: 59,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 558,
  confidence: 0.0836,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_010' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.4208,
  latency: 102,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 7381,
  confidence: 0.9009,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_011' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.571,
  latency: 219,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 892,
  confidence: 0.2106,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_012' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.3873,
  latency: 234,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 1677,
  confidence: 0.1342,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_013' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.5567,
  latency: 88,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 8334,
  confidence: 0.5498,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_014' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.546,
  latency: 216,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 8873,
  confidence: 0.5163,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_015' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.5457,
  latency: 224,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 4016,
  confidence: 0.0836,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_016' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.8131,
  latency: 198,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 8228,
  confidence: 0.2409,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_017' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.0229,
  latency: 107,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 8701,
  confidence: 0.9306,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_018' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_019' })
MERGE (a)-[r_018:CALIBRATES {
  strength: 0.9851,
  latency: 185,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 7051,
  confidence: 0.6268,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_019' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.0203,
  latency: 12,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 577,
  confidence: 0.2816,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_020' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.8712,
  latency: 175,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6229,
  confidence: 0.4506,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_021' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.5509,
  latency: 39,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 9075,
  confidence: 0.4976,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_022' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.3827,
  latency: 52,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 7310,
  confidence: 0.2479,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_023' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.2852,
  latency: 250,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 7940,
  confidence: 0.8659,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_024' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.838,
  latency: 250,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 2184,
  confidence: 0.3254,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_025' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.5844,
  latency: 248,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1157,
  confidence: 0.919,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_026' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.3209,
  latency: 4,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6286,
  confidence: 0.2236,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_027' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_028' })
MERGE (a)-[r_027:CALIBRATES {
  strength: 0.7661,
  latency: 13,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 7169,
  confidence: 0.9935,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_028' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.2555,
  latency: 203,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 5867,
  confidence: 0.311,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_01_core_engine_2_029' }),
      (b:Compression { identifier: 'compression_01_core_engine_2_000' })
MERGE (a)-[r_029:VALIDATES {
  strength: 0.6076,
  latency: 208,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 6671,
  confidence: 0.4495,
  active: true
}]->(b);
