:param namespace => 'transformer_02_02';
:param batchSize => 256;
:param threshold => 0.652;
:param maxDepth => 6;
:param timeoutSeconds => 94;
:param region => 'eu-west';
:param epoch => 80;
:param version => '4.4.7';

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_000' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.8083,
  latency: 55,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 8986,
  confidence: 0.8678,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_001' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.881,
  latency: 13,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 694,
  confidence: 0.0917,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_002' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.4507,
  latency: 154,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 8403,
  confidence: 0.7841,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_003' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.2721,
  latency: 119,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 4146,
  confidence: 0.8426,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_004' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.5734,
  latency: 150,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 8563,
  confidence: 0.9524,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_005' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.6136,
  latency: 56,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 863,
  confidence: 0.8475,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_006' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.3224,
  latency: 150,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 9455,
  confidence: 0.2454,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_007' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.6175,
  latency: 226,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 5275,
  confidence: 0.7891,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_008' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.3158,
  latency: 176,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 2687,
  confidence: 0.7875,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_009' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.9422,
  latency: 205,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 8428,
  confidence: 0.962,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_010' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.2482,
  latency: 241,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 6058,
  confidence: 0.4599,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_011' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.0211,
  latency: 217,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 7068,
  confidence: 0.6355,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_012' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.7676,
  latency: 67,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 6770,
  confidence: 0.6181,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_013' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.5012,
  latency: 104,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 5344,
  confidence: 0.0347,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_014' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.224,
  latency: 177,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 5563,
  confidence: 0.1418,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_015' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.9147,
  latency: 173,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 4257,
  confidence: 0.2067,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_016' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.3348,
  latency: 98,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 7629,
  confidence: 0.2401,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_017' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.6686,
  latency: 73,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 7852,
  confidence: 0.0386,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_018' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.9145,
  latency: 81,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 674,
  confidence: 0.46,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_019' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.5797,
  latency: 120,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 3812,
  confidence: 0.8399,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_020' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.6001,
  latency: 212,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 5489,
  confidence: 0.9862,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_021' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.7634,
  latency: 181,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 1141,
  confidence: 0.2612,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_022' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.0045,
  latency: 242,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 9555,
  confidence: 0.8507,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_023' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.2121,
  latency: 143,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 8444,
  confidence: 0.2656,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_024' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.1524,
  latency: 19,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 6273,
  confidence: 0.5831,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_025' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.3834,
  latency: 139,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 2329,
  confidence: 0.3206,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_026' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.94,
  latency: 124,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 5772,
  confidence: 0.9151,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_027' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.1135,
  latency: 61,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 5855,
  confidence: 0.5722,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_028' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.6488,
  latency: 233,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 7416,
  confidence: 0.6052,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_06_validation_layer_2_029' }),
      (b:Transformer { identifier: 'transformer_06_validation_layer_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.2293,
  latency: 89,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 6602,
  confidence: 0.4007,
  active: true
}]->(b);
