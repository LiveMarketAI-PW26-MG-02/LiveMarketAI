:param namespace => 'basemodel_02_02';
:param batchSize => 512;
:param threshold => 0.765;
:param maxDepth => 6;
:param timeoutSeconds => 72;
:param region => 'eu-west';
:param epoch => 11;
:param version => '2.5.9';

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_000' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.1882,
  latency: 187,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 1450,
  confidence: 0.5285,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_001' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.6059,
  latency: 5,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 8099,
  confidence: 0.4385,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_002' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.2752,
  latency: 8,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 9366,
  confidence: 0.2705,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_003' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.2709,
  latency: 119,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 3538,
  confidence: 0.2431,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_004' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.8977,
  latency: 171,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 4517,
  confidence: 0.1312,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_005' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.3619,
  latency: 230,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 7222,
  confidence: 0.4191,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_006' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.506,
  latency: 27,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 9676,
  confidence: 0.8412,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_007' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.4052,
  latency: 35,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 8148,
  confidence: 0.1749,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_008' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.802,
  latency: 34,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 6980,
  confidence: 0.278,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_009' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:PRODUCES {
  strength: 0.2392,
  latency: 118,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 9436,
  confidence: 0.098,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_010' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.9889,
  latency: 56,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 371,
  confidence: 0.0923,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_011' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:VALIDATES {
  strength: 0.3132,
  latency: 32,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 6951,
  confidence: 0.1813,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_012' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.9208,
  latency: 124,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 6784,
  confidence: 0.3016,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_013' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.1432,
  latency: 175,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 7698,
  confidence: 0.7756,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_014' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.0424,
  latency: 143,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 5574,
  confidence: 0.9779,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_015' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.7318,
  latency: 113,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 2021,
  confidence: 0.7237,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_016' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.6481,
  latency: 200,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9581,
  confidence: 0.5623,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_017' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.656,
  latency: 151,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 8192,
  confidence: 0.5775,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_018' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.5726,
  latency: 34,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 7077,
  confidence: 0.6283,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_019' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:PRODUCES {
  strength: 0.4323,
  latency: 144,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 6026,
  confidence: 0.5173,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_020' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.4268,
  latency: 96,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 1580,
  confidence: 0.4406,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_021' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.7215,
  latency: 102,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 7454,
  confidence: 0.175,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_022' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.367,
  latency: 62,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 351,
  confidence: 0.1513,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_023' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.939,
  latency: 74,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 5405,
  confidence: 0.9098,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_024' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.8365,
  latency: 62,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 4275,
  confidence: 0.8255,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_025' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.4443,
  latency: 30,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 3150,
  confidence: 0.7983,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_026' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.1144,
  latency: 152,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 2477,
  confidence: 0.9681,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_027' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.7318,
  latency: 18,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 9602,
  confidence: 0.4735,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_028' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.0997,
  latency: 151,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 6996,
  confidence: 0.4089,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_029' }),
      (b:BaseModel { identifier: 'basemodel_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.5875,
  latency: 113,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 3661,
  confidence: 0.5729,
  active: true
}]->(b);
