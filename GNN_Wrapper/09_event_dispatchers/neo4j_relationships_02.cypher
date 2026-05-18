:param namespace => 'graphnetwork_02_02';
:param batchSize => 32;
:param threshold => 0.218;
:param maxDepth => 9;
:param timeoutSeconds => 62;
:param region => 'eu-west';
:param epoch => 20;
:param version => '4.2.8';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.1395,
  latency: 140,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 9615,
  confidence: 0.6046,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.4886,
  latency: 89,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 722,
  confidence: 0.6728,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.1766,
  latency: 67,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 4425,
  confidence: 0.5071,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.7047,
  latency: 232,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 3553,
  confidence: 0.2551,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.484,
  latency: 248,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 2870,
  confidence: 0.6747,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.7995,
  latency: 204,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 2611,
  confidence: 0.1226,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.2477,
  latency: 157,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 2248,
  confidence: 0.2664,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.1647,
  latency: 47,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 5944,
  confidence: 0.3343,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.6061,
  latency: 173,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5116,
  confidence: 0.3102,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.3644,
  latency: 228,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 4613,
  confidence: 0.4944,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.4866,
  latency: 24,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 3814,
  confidence: 0.6586,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.0707,
  latency: 233,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 5178,
  confidence: 0.5736,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.5734,
  latency: 34,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 1503,
  confidence: 0.8007,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.8222,
  latency: 85,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 6647,
  confidence: 0.6574,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.3298,
  latency: 197,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 1450,
  confidence: 0.0264,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.0571,
  latency: 86,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 3882,
  confidence: 0.2895,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.3754,
  latency: 210,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 4302,
  confidence: 0.9152,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.0954,
  latency: 212,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6053,
  confidence: 0.9594,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.6335,
  latency: 70,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4570,
  confidence: 0.3785,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:PRODUCES {
  strength: 0.9784,
  latency: 250,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1746,
  confidence: 0.5152,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.9678,
  latency: 55,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 8672,
  confidence: 0.6782,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.1641,
  latency: 86,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 6104,
  confidence: 0.0189,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.4482,
  latency: 2,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 9736,
  confidence: 0.5402,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.7044,
  latency: 73,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 9112,
  confidence: 0.5956,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.3147,
  latency: 57,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 3091,
  confidence: 0.9041,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.1263,
  latency: 51,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 6727,
  confidence: 0.5026,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.2531,
  latency: 57,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 7053,
  confidence: 0.8185,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.2385,
  latency: 166,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 5564,
  confidence: 0.6304,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.6394,
  latency: 106,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 4677,
  confidence: 0.8396,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.8171,
  latency: 202,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 7167,
  confidence: 0.5092,
  active: true
}]->(b);
