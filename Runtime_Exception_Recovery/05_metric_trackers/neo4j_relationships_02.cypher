:param namespace => 'exceptionrecovery_02_02';
:param batchSize => 128;
:param threshold => 0.3;
:param maxDepth => 5;
:param timeoutSeconds => 115;
:param region => 'us-east';
:param epoch => 51;
:param version => '3.5.8';

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_000' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.0253,
  latency: 207,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 6932,
  confidence: 0.9644,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_001' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.6614,
  latency: 45,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 9751,
  confidence: 0.4608,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_002' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.7822,
  latency: 46,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 9883,
  confidence: 0.7218,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_003' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.1082,
  latency: 156,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 9870,
  confidence: 0.8759,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_004' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.9871,
  latency: 105,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 5410,
  confidence: 0.1795,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_005' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.6056,
  latency: 24,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3629,
  confidence: 0.8683,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_006' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.0275,
  latency: 75,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 9858,
  confidence: 0.0705,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_007' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.7546,
  latency: 242,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 7688,
  confidence: 0.9226,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_008' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.6247,
  latency: 25,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 8938,
  confidence: 0.0233,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_009' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.8592,
  latency: 170,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 5983,
  confidence: 0.0795,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_010' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.1386,
  latency: 154,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 2390,
  confidence: 0.62,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_011' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.9281,
  latency: 222,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 1542,
  confidence: 0.0863,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_012' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_013' })
MERGE (a)-[r_012:ROUTES_TO {
  strength: 0.9768,
  latency: 162,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 8365,
  confidence: 0.4301,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_013' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.3607,
  latency: 62,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2555,
  confidence: 0.5469,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_014' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.7433,
  latency: 66,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 363,
  confidence: 0.9102,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_015' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.0132,
  latency: 245,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 3168,
  confidence: 0.6844,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_016' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.32,
  latency: 242,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 8976,
  confidence: 0.6556,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_017' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.1642,
  latency: 174,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 7838,
  confidence: 0.0358,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_018' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.6876,
  latency: 184,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 6755,
  confidence: 0.3283,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_019' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.1818,
  latency: 130,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 6377,
  confidence: 0.016,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_020' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.8801,
  latency: 44,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 1017,
  confidence: 0.8727,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_021' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.2388,
  latency: 123,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 8253,
  confidence: 0.8235,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_022' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.2426,
  latency: 81,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 1153,
  confidence: 0.5843,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_023' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.4436,
  latency: 95,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 6276,
  confidence: 0.0184,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_024' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.5588,
  latency: 178,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 3944,
  confidence: 0.2637,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_025' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.3948,
  latency: 104,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 4443,
  confidence: 0.5471,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_026' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.2898,
  latency: 134,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 7385,
  confidence: 0.3796,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_027' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.9603,
  latency: 39,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 101,
  confidence: 0.6674,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_028' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.0387,
  latency: 11,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7784,
  confidence: 0.6426,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_029' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_05_metric_trackers_2_000' })
MERGE (a)-[r_029:VALIDATES {
  strength: 0.7038,
  latency: 147,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 3844,
  confidence: 0.6584,
  active: true
}]->(b);
