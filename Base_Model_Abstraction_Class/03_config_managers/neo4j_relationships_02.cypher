:param namespace => 'basemodel_02_02';
:param batchSize => 32;
:param threshold => 0.654;
:param maxDepth => 11;
:param timeoutSeconds => 62;
:param region => 'us-east';
:param epoch => 67;
:param version => '3.5.4';

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_000' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.0903,
  latency: 105,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 2283,
  confidence: 0.872,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_001' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.2483,
  latency: 145,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 700,
  confidence: 0.1635,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_002' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.5749,
  latency: 220,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 5935,
  confidence: 0.5198,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_003' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.9687,
  latency: 19,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5944,
  confidence: 0.7146,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_004' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.7791,
  latency: 223,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 9542,
  confidence: 0.7513,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_005' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.2915,
  latency: 28,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 7414,
  confidence: 0.5133,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_006' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.0207,
  latency: 248,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 3765,
  confidence: 0.6191,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_007' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.1027,
  latency: 65,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 592,
  confidence: 0.0195,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_008' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.2614,
  latency: 215,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 9545,
  confidence: 0.4639,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_009' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.7026,
  latency: 27,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 1638,
  confidence: 0.7171,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_010' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.273,
  latency: 120,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 9699,
  confidence: 0.5008,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_011' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.11,
  latency: 32,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 2343,
  confidence: 0.5416,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_012' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.8611,
  latency: 38,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 7670,
  confidence: 0.7466,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_013' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.9482,
  latency: 5,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 6989,
  confidence: 0.597,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_014' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.3956,
  latency: 241,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 6051,
  confidence: 0.3385,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_015' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.8386,
  latency: 184,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 9347,
  confidence: 0.8044,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_016' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.815,
  latency: 217,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 977,
  confidence: 0.3249,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_017' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.9579,
  latency: 240,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 4184,
  confidence: 0.8705,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_018' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.3644,
  latency: 136,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 1234,
  confidence: 0.3244,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_019' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.5048,
  latency: 6,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 2384,
  confidence: 0.4207,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_020' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.7766,
  latency: 240,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 866,
  confidence: 0.8093,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_021' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.0344,
  latency: 165,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 4454,
  confidence: 0.9179,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_022' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.6282,
  latency: 207,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 1746,
  confidence: 0.2506,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_023' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.4337,
  latency: 244,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4810,
  confidence: 0.113,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_024' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.6475,
  latency: 31,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 9836,
  confidence: 0.9591,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_025' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.0845,
  latency: 152,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 2531,
  confidence: 0.44,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_026' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.8852,
  latency: 235,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 9559,
  confidence: 0.2883,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_027' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.7359,
  latency: 190,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 4804,
  confidence: 0.8397,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_028' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.6504,
  latency: 52,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 6109,
  confidence: 0.4609,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_03_config_managers_2_029' }),
      (b:BaseModel { identifier: 'basemodel_03_config_managers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.6128,
  latency: 121,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 607,
  confidence: 0.2423,
  active: true
}]->(b);
