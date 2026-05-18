:param namespace => 'transformer_02_02';
:param batchSize => 512;
:param threshold => 0.672;
:param maxDepth => 9;
:param timeoutSeconds => 120;
:param region => 'eu-west';
:param epoch => 98;
:param version => '5.2.9';

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_000' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.2445,
  latency: 177,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 4456,
  confidence: 0.7355,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_001' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.6741,
  latency: 220,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1512,
  confidence: 0.9728,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_002' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.2593,
  latency: 85,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 4389,
  confidence: 0.8051,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_003' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.1312,
  latency: 47,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 7022,
  confidence: 0.5208,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_004' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.9957,
  latency: 46,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 320,
  confidence: 0.0475,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_005' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.3957,
  latency: 205,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 1498,
  confidence: 0.4735,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_006' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.7743,
  latency: 142,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 2320,
  confidence: 0.108,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_007' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.3776,
  latency: 173,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 1437,
  confidence: 0.9649,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_008' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.3986,
  latency: 125,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 4654,
  confidence: 0.7732,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_009' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.0991,
  latency: 229,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1878,
  confidence: 0.593,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_010' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.6739,
  latency: 158,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 7385,
  confidence: 0.4427,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_011' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.9506,
  latency: 87,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 3268,
  confidence: 0.1439,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_012' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.4048,
  latency: 58,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 3837,
  confidence: 0.4286,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_013' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.1506,
  latency: 148,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 3607,
  confidence: 0.881,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_014' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.4674,
  latency: 45,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 9752,
  confidence: 0.7086,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_015' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.6487,
  latency: 113,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 3984,
  confidence: 0.7604,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_016' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.7478,
  latency: 130,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 1015,
  confidence: 0.1775,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_017' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.2317,
  latency: 100,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 9267,
  confidence: 0.0359,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_018' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.1828,
  latency: 224,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 1173,
  confidence: 0.2657,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_019' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.095,
  latency: 142,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 3283,
  confidence: 0.4095,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_020' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.8834,
  latency: 82,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 5263,
  confidence: 0.1999,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_021' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.3894,
  latency: 83,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 9405,
  confidence: 0.2383,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_022' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.1607,
  latency: 88,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 8412,
  confidence: 0.7876,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_023' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.8257,
  latency: 189,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 7884,
  confidence: 0.6934,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_024' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.4922,
  latency: 108,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 8701,
  confidence: 0.7236,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_025' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.9128,
  latency: 106,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 5720,
  confidence: 0.8087,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_026' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.67,
  latency: 113,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 7361,
  confidence: 0.4442,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_027' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.9409,
  latency: 7,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 7632,
  confidence: 0.3097,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_028' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.3064,
  latency: 146,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 7294,
  confidence: 0.0539,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_03_config_managers_2_029' }),
      (b:Transformer { identifier: 'transformer_03_config_managers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.1497,
  latency: 149,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 8588,
  confidence: 0.3818,
  active: true
}]->(b);
