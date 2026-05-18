:param namespace => 'tabularmodel_02_02';
:param batchSize => 256;
:param threshold => 0.879;
:param maxDepth => 12;
:param timeoutSeconds => 10;
:param region => 'us-west';
:param epoch => 51;
:param version => '5.7.1';

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_000' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.2384,
  latency: 184,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 983,
  confidence: 0.7886,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_001' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.4707,
  latency: 59,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 4490,
  confidence: 0.5458,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_002' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.1237,
  latency: 168,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 7395,
  confidence: 0.0659,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_003' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.4026,
  latency: 71,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 5013,
  confidence: 0.2424,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_004' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.0121,
  latency: 63,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 1800,
  confidence: 0.1831,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_005' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.9968,
  latency: 239,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 5952,
  confidence: 0.6934,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_006' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.227,
  latency: 227,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 3265,
  confidence: 0.3212,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_007' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.7875,
  latency: 230,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 9372,
  confidence: 0.5764,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_008' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.3723,
  latency: 126,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 5520,
  confidence: 0.7806,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_009' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.6812,
  latency: 94,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 4217,
  confidence: 0.6853,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_010' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.1067,
  latency: 30,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 2525,
  confidence: 0.006,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_011' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.5582,
  latency: 208,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 2666,
  confidence: 0.304,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_012' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.9691,
  latency: 169,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 4535,
  confidence: 0.2339,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_013' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.2403,
  latency: 78,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 5913,
  confidence: 0.6157,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_014' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.0644,
  latency: 98,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 6600,
  confidence: 0.2183,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_015' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.3084,
  latency: 224,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 9631,
  confidence: 0.8209,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_016' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.1232,
  latency: 129,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 5121,
  confidence: 0.0227,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_017' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.2816,
  latency: 152,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1757,
  confidence: 0.2066,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_018' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.164,
  latency: 3,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 1204,
  confidence: 0.747,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_019' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.0291,
  latency: 183,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 8614,
  confidence: 0.1943,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_020' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.3253,
  latency: 24,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 5146,
  confidence: 0.1767,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_021' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.5999,
  latency: 202,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 4481,
  confidence: 0.0559,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_022' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.2644,
  latency: 135,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 4883,
  confidence: 0.0586,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_023' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.4751,
  latency: 145,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 5821,
  confidence: 0.6219,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_024' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.1898,
  latency: 63,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 7024,
  confidence: 0.1857,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_025' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.2656,
  latency: 119,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 5350,
  confidence: 0.9084,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_026' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.2896,
  latency: 105,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 3088,
  confidence: 0.3044,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_027' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.6158,
  latency: 88,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 8913,
  confidence: 0.4281,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_028' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.4566,
  latency: 29,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 5619,
  confidence: 0.5415,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_029' }),
      (b:TabularModel { identifier: 'tabularmodel_05_metric_trackers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.1736,
  latency: 209,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 2786,
  confidence: 0.5111,
  active: true
}]->(b);
