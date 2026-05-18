:param namespace => 'exceptionrecovery_02_02';
:param batchSize => 512;
:param threshold => 0.603;
:param maxDepth => 8;
:param timeoutSeconds => 98;
:param region => 'us-east';
:param epoch => 92;
:param version => '3.5.8';

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_000' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.14,
  latency: 132,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 1721,
  confidence: 0.239,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_001' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.0508,
  latency: 107,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 3279,
  confidence: 0.1263,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_002' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.2736,
  latency: 114,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 4868,
  confidence: 0.6353,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_003' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.5924,
  latency: 229,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 3321,
  confidence: 0.0838,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_004' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.4743,
  latency: 65,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 2171,
  confidence: 0.0127,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_005' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.2917,
  latency: 101,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 6167,
  confidence: 0.7912,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_006' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.4819,
  latency: 241,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 2343,
  confidence: 0.5717,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_007' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.098,
  latency: 234,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 6147,
  confidence: 0.6013,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_008' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.316,
  latency: 98,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 3906,
  confidence: 0.7582,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_009' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.8065,
  latency: 186,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 2510,
  confidence: 0.3026,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_010' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.6416,
  latency: 80,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 5156,
  confidence: 0.0358,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_011' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.2414,
  latency: 40,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 8212,
  confidence: 0.025,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_012' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.9802,
  latency: 22,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 9833,
  confidence: 0.3726,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_013' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.9196,
  latency: 17,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 4386,
  confidence: 0.0623,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_014' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:TRIGGERS {
  strength: 0.7428,
  latency: 120,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 8363,
  confidence: 0.5295,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_015' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.1254,
  latency: 152,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 8796,
  confidence: 0.8946,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_016' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.0335,
  latency: 105,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 9978,
  confidence: 0.5065,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_017' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.5206,
  latency: 141,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 5597,
  confidence: 0.5361,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_018' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.7979,
  latency: 222,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 7322,
  confidence: 0.0906,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_019' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:CALIBRATES {
  strength: 0.6201,
  latency: 184,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 5691,
  confidence: 0.817,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_020' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.6168,
  latency: 40,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 3976,
  confidence: 0.8513,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_021' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.2719,
  latency: 65,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 7845,
  confidence: 0.9498,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_022' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.8005,
  latency: 69,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 1548,
  confidence: 0.8437,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_023' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.6891,
  latency: 90,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 9126,
  confidence: 0.8384,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_024' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.865,
  latency: 64,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 7511,
  confidence: 0.4709,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_025' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.6844,
  latency: 103,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 9734,
  confidence: 0.0603,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_026' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.2233,
  latency: 34,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 6556,
  confidence: 0.4307,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_027' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.4271,
  latency: 209,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 9964,
  confidence: 0.6532,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_028' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.1415,
  latency: 239,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 5378,
  confidence: 0.5895,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_029' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.562,
  latency: 90,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 9720,
  confidence: 0.072,
  active: true
}]->(b);
