:param namespace => 'exceptionrecovery_02_02';
:param batchSize => 128;
:param threshold => 0.172;
:param maxDepth => 4;
:param timeoutSeconds => 23;
:param region => 'us-west';
:param epoch => 44;
:param version => '3.6.8';

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_000' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.9017,
  latency: 166,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 261,
  confidence: 0.0445,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_001' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.0026,
  latency: 10,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 3139,
  confidence: 0.6836,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_002' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.5573,
  latency: 119,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 5374,
  confidence: 0.0039,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_003' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.4384,
  latency: 146,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 3828,
  confidence: 0.6839,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_004' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.5405,
  latency: 176,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 3390,
  confidence: 0.9377,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_005' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.6884,
  latency: 56,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 8949,
  confidence: 0.3955,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_006' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.3772,
  latency: 88,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 4166,
  confidence: 0.7325,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_007' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.0847,
  latency: 232,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 625,
  confidence: 0.1665,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_008' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.2759,
  latency: 216,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 3907,
  confidence: 0.9467,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_009' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.131,
  latency: 218,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 193,
  confidence: 0.5571,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_010' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.1828,
  latency: 228,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 935,
  confidence: 0.783,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_011' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.791,
  latency: 193,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 9277,
  confidence: 0.78,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_012' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.6838,
  latency: 77,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 7564,
  confidence: 0.5652,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_013' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.0923,
  latency: 48,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 298,
  confidence: 0.5827,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_014' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.123,
  latency: 46,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 163,
  confidence: 0.8856,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_015' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.5855,
  latency: 166,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4456,
  confidence: 0.0448,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_016' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.9223,
  latency: 224,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 4506,
  confidence: 0.9164,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_017' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.001,
  latency: 152,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 9797,
  confidence: 0.1246,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_018' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.156,
  latency: 75,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 1534,
  confidence: 0.2177,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_019' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.8791,
  latency: 49,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 9326,
  confidence: 0.1589,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_020' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.1336,
  latency: 232,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 237,
  confidence: 0.1422,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_021' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.2884,
  latency: 124,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 639,
  confidence: 0.1961,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_022' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.8527,
  latency: 247,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 9975,
  confidence: 0.0312,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_023' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.1387,
  latency: 40,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 3560,
  confidence: 0.4449,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_024' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.6713,
  latency: 75,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 2427,
  confidence: 0.7715,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_025' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.3245,
  latency: 21,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 6560,
  confidence: 0.7038,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_026' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.0839,
  latency: 168,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 6053,
  confidence: 0.1042,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_027' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.2698,
  latency: 199,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 8612,
  confidence: 0.6109,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_028' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.3664,
  latency: 242,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 3087,
  confidence: 0.7667,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_029' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_01_core_engine_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.4237,
  latency: 223,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 5460,
  confidence: 0.0316,
  active: true
}]->(b);
