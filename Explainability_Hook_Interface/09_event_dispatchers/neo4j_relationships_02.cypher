:param namespace => 'explainability_02_02';
:param batchSize => 32;
:param threshold => 0.137;
:param maxDepth => 12;
:param timeoutSeconds => 43;
:param region => 'us-east';
:param epoch => 94;
:param version => '1.3.4';

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_000' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.1375,
  latency: 14,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 9824,
  confidence: 0.0102,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_001' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.9709,
  latency: 50,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 3988,
  confidence: 0.158,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_002' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.6542,
  latency: 233,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 7496,
  confidence: 0.3958,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_003' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.496,
  latency: 8,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 8769,
  confidence: 0.5166,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_004' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.8026,
  latency: 118,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 2578,
  confidence: 0.0909,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_005' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.7391,
  latency: 215,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 2785,
  confidence: 0.8039,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_006' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.6826,
  latency: 45,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 8595,
  confidence: 0.4416,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_007' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.6016,
  latency: 228,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 9046,
  confidence: 0.401,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_008' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.8005,
  latency: 9,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 1937,
  confidence: 0.2223,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_009' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.5186,
  latency: 233,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 5858,
  confidence: 0.7451,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_010' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.8855,
  latency: 47,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 1762,
  confidence: 0.9658,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_011' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.8716,
  latency: 124,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 3601,
  confidence: 0.0529,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_012' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.4315,
  latency: 159,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6982,
  confidence: 0.9599,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_013' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:MONITORS {
  strength: 0.1067,
  latency: 182,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 4714,
  confidence: 0.87,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_014' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.0755,
  latency: 248,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 8295,
  confidence: 0.5904,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_015' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.5869,
  latency: 239,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 2009,
  confidence: 0.8583,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_016' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.8576,
  latency: 114,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 8449,
  confidence: 0.4432,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_017' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.7387,
  latency: 156,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 1221,
  confidence: 0.4156,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_018' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.8696,
  latency: 226,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 9236,
  confidence: 0.6485,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_019' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.3578,
  latency: 8,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 7547,
  confidence: 0.8218,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_020' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.5472,
  latency: 222,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 3007,
  confidence: 0.669,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_021' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.156,
  latency: 183,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 1332,
  confidence: 0.7238,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_022' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.3343,
  latency: 217,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 2012,
  confidence: 0.4533,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_023' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:ROUTES_TO {
  strength: 0.3732,
  latency: 98,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 865,
  confidence: 0.1382,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_024' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.8851,
  latency: 126,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 5349,
  confidence: 0.3141,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_025' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.032,
  latency: 74,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 2338,
  confidence: 0.0481,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_026' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.9273,
  latency: 32,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 9777,
  confidence: 0.912,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_027' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.937,
  latency: 202,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5515,
  confidence: 0.4049,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_028' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.2283,
  latency: 89,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 841,
  confidence: 0.0897,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_09_event_dispatchers_2_029' }),
      (b:Explainability { identifier: 'explainability_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.5682,
  latency: 123,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 7137,
  confidence: 0.1093,
  active: true
}]->(b);
