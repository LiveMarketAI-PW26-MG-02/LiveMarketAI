:param namespace => 'basemodel_02_02';
:param batchSize => 128;
:param threshold => 0.597;
:param maxDepth => 3;
:param timeoutSeconds => 77;
:param region => 'eu-west';
:param epoch => 15;
:param version => '2.4.0';

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_000' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.2018,
  latency: 80,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 8801,
  confidence: 0.7595,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_001' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.4457,
  latency: 173,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 4532,
  confidence: 0.347,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_002' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.9945,
  latency: 10,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 402,
  confidence: 0.7331,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_003' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.5142,
  latency: 63,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 1841,
  confidence: 0.6583,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_004' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.6565,
  latency: 140,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8401,
  confidence: 0.3078,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_005' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.9824,
  latency: 88,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 2389,
  confidence: 0.4047,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_006' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.9819,
  latency: 215,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 333,
  confidence: 0.0707,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_007' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.4307,
  latency: 15,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6340,
  confidence: 0.8705,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_008' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.5988,
  latency: 178,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 841,
  confidence: 0.4595,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_009' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.269,
  latency: 1,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 6066,
  confidence: 0.9618,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_010' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.2444,
  latency: 248,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 3669,
  confidence: 0.3566,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_011' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_012' })
MERGE (a)-[r_011:DEPENDS_ON {
  strength: 0.3353,
  latency: 22,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 4669,
  confidence: 0.5028,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_012' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.2482,
  latency: 199,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 1588,
  confidence: 0.2642,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_013' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.1439,
  latency: 151,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 6554,
  confidence: 0.0225,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_014' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.6297,
  latency: 22,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 8770,
  confidence: 0.8532,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_015' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.6575,
  latency: 184,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 6481,
  confidence: 0.7643,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_016' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.1495,
  latency: 186,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 2471,
  confidence: 0.0438,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_017' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.7339,
  latency: 208,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 2382,
  confidence: 0.9099,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_018' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.8264,
  latency: 150,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 1494,
  confidence: 0.0312,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_019' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.6371,
  latency: 246,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6270,
  confidence: 0.8358,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_020' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.6278,
  latency: 161,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 4106,
  confidence: 0.4893,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_021' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.4569,
  latency: 18,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 8868,
  confidence: 0.0919,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_022' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.7457,
  latency: 122,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 1319,
  confidence: 0.8461,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_023' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.7293,
  latency: 53,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 7642,
  confidence: 0.4939,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_024' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.0767,
  latency: 234,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 865,
  confidence: 0.617,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_025' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.0775,
  latency: 38,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 4260,
  confidence: 0.6515,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_026' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.6212,
  latency: 35,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 8003,
  confidence: 0.0607,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_027' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.9725,
  latency: 26,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 8121,
  confidence: 0.2909,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_028' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_029' })
MERGE (a)-[r_028:ROUTES_TO {
  strength: 0.4647,
  latency: 120,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 9096,
  confidence: 0.1993,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_01_core_engine_2_029' }),
      (b:BaseModel { identifier: 'basemodel_01_core_engine_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.9363,
  latency: 5,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 7619,
  confidence: 0.0765,
  active: true
}]->(b);
