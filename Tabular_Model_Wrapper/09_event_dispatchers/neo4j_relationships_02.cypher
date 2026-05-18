:param namespace => 'tabularmodel_02_02';
:param batchSize => 512;
:param threshold => 0.615;
:param maxDepth => 5;
:param timeoutSeconds => 60;
:param region => 'us-east';
:param epoch => 78;
:param version => '4.8.0';

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_000' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.3552,
  latency: 49,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 7261,
  confidence: 0.0225,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_001' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.5985,
  latency: 57,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 4727,
  confidence: 0.5047,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_002' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.6418,
  latency: 73,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 8181,
  confidence: 0.2681,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_003' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.3632,
  latency: 40,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 6116,
  confidence: 0.1573,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_004' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.0051,
  latency: 84,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 7490,
  confidence: 0.5935,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_005' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.714,
  latency: 50,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 7098,
  confidence: 0.4846,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_006' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.8567,
  latency: 69,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 9579,
  confidence: 0.1409,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_007' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.9062,
  latency: 224,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 9304,
  confidence: 0.8636,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_008' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.8716,
  latency: 165,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 1209,
  confidence: 0.9997,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_009' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.4543,
  latency: 174,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 447,
  confidence: 0.6938,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_010' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.2465,
  latency: 195,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 7326,
  confidence: 0.8702,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_011' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.0275,
  latency: 230,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 2317,
  confidence: 0.4763,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_012' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.3837,
  latency: 134,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 4421,
  confidence: 0.1931,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_013' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.7778,
  latency: 45,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 8426,
  confidence: 0.1747,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_014' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.2424,
  latency: 29,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 276,
  confidence: 0.4558,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_015' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.392,
  latency: 22,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 2335,
  confidence: 0.0264,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_016' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.4188,
  latency: 141,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 3693,
  confidence: 0.5722,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_017' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.7937,
  latency: 215,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 8409,
  confidence: 0.3108,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_018' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.0815,
  latency: 141,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 9819,
  confidence: 0.6386,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_019' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.1227,
  latency: 99,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 9203,
  confidence: 0.3238,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_020' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.5962,
  latency: 248,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 7186,
  confidence: 0.9213,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_021' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.4225,
  latency: 13,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 9213,
  confidence: 0.0673,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_022' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.8975,
  latency: 136,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 3390,
  confidence: 0.0279,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_023' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.5442,
  latency: 133,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 5896,
  confidence: 0.4604,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_024' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.9306,
  latency: 99,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 1337,
  confidence: 0.5284,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_025' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.462,
  latency: 64,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 5851,
  confidence: 0.7877,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_026' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.9775,
  latency: 74,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 2815,
  confidence: 0.7929,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_027' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.3253,
  latency: 146,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 9247,
  confidence: 0.1347,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_028' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.6073,
  latency: 171,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 1075,
  confidence: 0.713,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_029' }),
      (b:TabularModel { identifier: 'tabularmodel_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.3812,
  latency: 30,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 1210,
  confidence: 0.3079,
  active: true
}]->(b);
