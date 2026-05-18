:param namespace => 'exceptionrecovery_02_02';
:param batchSize => 256;
:param threshold => 0.311;
:param maxDepth => 9;
:param timeoutSeconds => 101;
:param region => 'ap-south';
:param epoch => 96;
:param version => '1.9.7';

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_000' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.088,
  latency: 175,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 4624,
  confidence: 0.933,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_001' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.6015,
  latency: 115,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 2807,
  confidence: 0.8766,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_002' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.5388,
  latency: 197,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 9667,
  confidence: 0.2473,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_003' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.726,
  latency: 229,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 8779,
  confidence: 0.1825,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_004' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.3189,
  latency: 190,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 9349,
  confidence: 0.9104,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_005' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.5004,
  latency: 211,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3544,
  confidence: 0.1636,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_006' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.0269,
  latency: 137,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 197,
  confidence: 0.4781,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_007' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.3306,
  latency: 170,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 741,
  confidence: 0.3308,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_008' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.185,
  latency: 211,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 5075,
  confidence: 0.9706,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_009' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.5202,
  latency: 86,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 5596,
  confidence: 0.7526,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_010' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.7525,
  latency: 224,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 8732,
  confidence: 0.8236,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_011' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.2993,
  latency: 11,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1511,
  confidence: 0.0997,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_012' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.7389,
  latency: 8,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 3908,
  confidence: 0.9551,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_013' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.8207,
  latency: 29,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 2473,
  confidence: 0.0506,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_014' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.5164,
  latency: 4,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 1484,
  confidence: 0.8884,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_015' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.9696,
  latency: 219,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 1871,
  confidence: 0.4639,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_016' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.1242,
  latency: 214,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 3863,
  confidence: 0.9328,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_017' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.3229,
  latency: 159,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 6933,
  confidence: 0.9212,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_018' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.6955,
  latency: 195,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 2230,
  confidence: 0.4177,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_019' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.6065,
  latency: 125,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 753,
  confidence: 0.8053,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_020' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.9271,
  latency: 157,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 5411,
  confidence: 0.4302,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_021' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.6289,
  latency: 208,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 5561,
  confidence: 0.8156,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_022' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.2088,
  latency: 224,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 7465,
  confidence: 0.5377,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_023' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.4048,
  latency: 111,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 1119,
  confidence: 0.8616,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_024' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.2698,
  latency: 73,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 2379,
  confidence: 0.2994,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_025' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.7746,
  latency: 40,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 9050,
  confidence: 0.3493,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_026' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.3148,
  latency: 65,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 7010,
  confidence: 0.0788,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_027' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.3759,
  latency: 199,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 5394,
  confidence: 0.7758,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_028' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.0061,
  latency: 221,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 6350,
  confidence: 0.7911,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_029' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.7506,
  latency: 9,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 6331,
  confidence: 0.6184,
  active: true
}]->(b);
