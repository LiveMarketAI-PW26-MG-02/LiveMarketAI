:param namespace => 'exceptionrecovery_02_02';
:param batchSize => 128;
:param threshold => 0.819;
:param maxDepth => 3;
:param timeoutSeconds => 66;
:param region => 'us-east';
:param epoch => 47;
:param version => '2.7.6';

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_000' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.892,
  latency: 174,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 525,
  confidence: 0.8334,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_001' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.0615,
  latency: 159,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 4380,
  confidence: 0.7378,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_002' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.0049,
  latency: 165,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 9193,
  confidence: 0.6163,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_003' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.6894,
  latency: 208,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 2491,
  confidence: 0.068,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_004' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.1185,
  latency: 170,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 5721,
  confidence: 0.156,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_005' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.7011,
  latency: 37,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2352,
  confidence: 0.6094,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_006' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.8061,
  latency: 175,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 4947,
  confidence: 0.8862,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_007' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.0936,
  latency: 219,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 8080,
  confidence: 0.5091,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_008' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.6706,
  latency: 171,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 9767,
  confidence: 0.8547,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_009' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.5102,
  latency: 211,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 336,
  confidence: 0.069,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_010' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.7358,
  latency: 187,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 6335,
  confidence: 0.2778,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_011' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.7953,
  latency: 220,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5671,
  confidence: 0.4559,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_012' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_013' })
MERGE (a)-[r_012:ROUTES_TO {
  strength: 0.8577,
  latency: 193,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6321,
  confidence: 0.5686,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_013' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.6122,
  latency: 135,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 1529,
  confidence: 0.0145,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_014' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.0967,
  latency: 160,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 5730,
  confidence: 0.3912,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_015' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.4439,
  latency: 197,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 5046,
  confidence: 0.3059,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_016' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.1652,
  latency: 228,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 1229,
  confidence: 0.2525,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_017' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.5371,
  latency: 134,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 8741,
  confidence: 0.4279,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_018' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.4792,
  latency: 83,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 9294,
  confidence: 0.1961,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_019' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.4018,
  latency: 120,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 6054,
  confidence: 0.7099,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_020' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.4372,
  latency: 138,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 6852,
  confidence: 0.0109,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_021' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.3987,
  latency: 49,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 1177,
  confidence: 0.2253,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_022' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.8518,
  latency: 183,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 2995,
  confidence: 0.6413,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_023' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.7405,
  latency: 177,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 3031,
  confidence: 0.2049,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_024' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.2975,
  latency: 195,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9894,
  confidence: 0.2651,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_025' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.8447,
  latency: 97,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 8259,
  confidence: 0.1991,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_026' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.4549,
  latency: 221,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 9529,
  confidence: 0.2517,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_027' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_028' })
MERGE (a)-[r_027:CALIBRATES {
  strength: 0.7981,
  latency: 196,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 1535,
  confidence: 0.3898,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_028' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.5544,
  latency: 120,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 7886,
  confidence: 0.793,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_029' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_06_validation_layer_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.4592,
  latency: 38,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 3635,
  confidence: 0.5432,
  active: true
}]->(b);
