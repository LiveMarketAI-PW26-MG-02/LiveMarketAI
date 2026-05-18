:param namespace => 'basemodel_02_02';
:param batchSize => 512;
:param threshold => 0.651;
:param maxDepth => 5;
:param timeoutSeconds => 24;
:param region => 'us-east';
:param epoch => 94;
:param version => '1.4.0';

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_000' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.7048,
  latency: 158,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 8313,
  confidence: 0.74,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_001' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.1178,
  latency: 119,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 8080,
  confidence: 0.9672,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_002' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.1067,
  latency: 237,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 6327,
  confidence: 0.9688,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_003' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.4802,
  latency: 183,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 6541,
  confidence: 0.5191,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_004' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.8341,
  latency: 151,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 7455,
  confidence: 0.2625,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_005' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.1534,
  latency: 100,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 4624,
  confidence: 0.3614,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_006' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.4254,
  latency: 241,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 4000,
  confidence: 0.1228,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_007' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.4162,
  latency: 9,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 7380,
  confidence: 0.6635,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_008' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.9109,
  latency: 113,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 1776,
  confidence: 0.9241,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_009' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:PRODUCES {
  strength: 0.4051,
  latency: 130,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 6251,
  confidence: 0.3641,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_010' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.0887,
  latency: 7,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 8353,
  confidence: 0.2225,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_011' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.8151,
  latency: 142,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 8581,
  confidence: 0.0705,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_012' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:ROUTES_TO {
  strength: 0.8211,
  latency: 107,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 4226,
  confidence: 0.586,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_013' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.8396,
  latency: 13,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 1698,
  confidence: 0.5431,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_014' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.3053,
  latency: 15,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 1745,
  confidence: 0.4279,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_015' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.5876,
  latency: 185,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 8241,
  confidence: 0.2894,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_016' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.0214,
  latency: 117,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 5430,
  confidence: 0.2991,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_017' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.6385,
  latency: 131,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 1642,
  confidence: 0.8012,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_018' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:CALIBRATES {
  strength: 0.3404,
  latency: 95,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 5287,
  confidence: 0.5088,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_019' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.7195,
  latency: 96,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 6854,
  confidence: 0.9138,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_020' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.5951,
  latency: 154,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7214,
  confidence: 0.9433,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_021' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.9461,
  latency: 220,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 3442,
  confidence: 0.1349,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_022' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.8097,
  latency: 143,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 1402,
  confidence: 0.2573,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_023' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.3604,
  latency: 177,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 3278,
  confidence: 0.3992,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_024' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.7136,
  latency: 25,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 1811,
  confidence: 0.1844,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_025' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.0431,
  latency: 49,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 6506,
  confidence: 0.6852,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_026' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.3746,
  latency: 179,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 4782,
  confidence: 0.4023,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_027' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.5154,
  latency: 49,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 2407,
  confidence: 0.9649,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_028' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.9926,
  latency: 120,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 1436,
  confidence: 0.2407,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_029' }),
      (b:BaseModel { identifier: 'basemodel_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.715,
  latency: 241,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 5988,
  confidence: 0.8793,
  active: true
}]->(b);
