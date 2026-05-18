:param namespace => 'graphnetwork_02_02';
:param batchSize => 256;
:param threshold => 0.705;
:param maxDepth => 10;
:param timeoutSeconds => 30;
:param region => 'us-east';
:param epoch => 75;
:param version => '4.5.4';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.018,
  latency: 11,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 9256,
  confidence: 0.0279,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.9341,
  latency: 171,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 9349,
  confidence: 0.6427,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.5707,
  latency: 6,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 8553,
  confidence: 0.467,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.9,
  latency: 181,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 5630,
  confidence: 0.2075,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.2656,
  latency: 195,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 9739,
  confidence: 0.3999,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_006' })
MERGE (a)-[r_005:OBSERVES {
  strength: 0.948,
  latency: 147,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 1928,
  confidence: 0.5828,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.3312,
  latency: 250,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 1256,
  confidence: 0.348,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.2255,
  latency: 151,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 8771,
  confidence: 0.9515,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.9248,
  latency: 236,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9045,
  confidence: 0.1704,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.871,
  latency: 214,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 2107,
  confidence: 0.7133,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.2561,
  latency: 32,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 2937,
  confidence: 0.1851,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.7584,
  latency: 50,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 5385,
  confidence: 0.0264,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.7641,
  latency: 137,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 9611,
  confidence: 0.3357,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.0063,
  latency: 245,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2136,
  confidence: 0.0989,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.4312,
  latency: 96,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 4958,
  confidence: 0.424,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.206,
  latency: 145,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 8709,
  confidence: 0.2496,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.8782,
  latency: 214,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 135,
  confidence: 0.3695,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_018' })
MERGE (a)-[r_017:PRODUCES {
  strength: 0.4172,
  latency: 35,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 6723,
  confidence: 0.3723,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.5235,
  latency: 172,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 4503,
  confidence: 0.5023,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.6835,
  latency: 32,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 9724,
  confidence: 0.6691,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.3907,
  latency: 155,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 1160,
  confidence: 0.1003,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.1485,
  latency: 114,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 3855,
  confidence: 0.5543,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.0833,
  latency: 177,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 237,
  confidence: 0.6375,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.0328,
  latency: 151,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 4463,
  confidence: 0.4717,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.4688,
  latency: 56,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 5782,
  confidence: 0.4304,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.4799,
  latency: 59,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 2322,
  confidence: 0.9892,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.8835,
  latency: 52,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 3042,
  confidence: 0.5695,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.0376,
  latency: 131,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 8135,
  confidence: 0.0112,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.1553,
  latency: 17,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 8592,
  confidence: 0.9436,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_05_metric_trackers_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.9184,
  latency: 72,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 3951,
  confidence: 0.7052,
  active: true
}]->(b);
