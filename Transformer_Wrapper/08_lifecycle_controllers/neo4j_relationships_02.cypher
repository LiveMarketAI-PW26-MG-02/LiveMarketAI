:param namespace => 'transformer_02_02';
:param batchSize => 128;
:param threshold => 0.74;
:param maxDepth => 6;
:param timeoutSeconds => 18;
:param region => 'ap-south';
:param epoch => 82;
:param version => '2.6.8';

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_000' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.3269,
  latency: 111,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9215,
  confidence: 0.6676,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_001' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.253,
  latency: 25,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 9231,
  confidence: 0.8888,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_002' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.8225,
  latency: 112,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 6441,
  confidence: 0.2363,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_003' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.4071,
  latency: 6,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 4816,
  confidence: 0.2779,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_004' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.3348,
  latency: 188,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 4386,
  confidence: 0.4213,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_005' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.4052,
  latency: 68,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 6941,
  confidence: 0.3703,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_006' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.3316,
  latency: 246,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 1725,
  confidence: 0.0366,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_007' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.7336,
  latency: 15,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 4116,
  confidence: 0.2863,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_008' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.0798,
  latency: 240,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 633,
  confidence: 0.1894,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_009' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.0271,
  latency: 226,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 4350,
  confidence: 0.6015,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_010' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.2177,
  latency: 173,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6752,
  confidence: 0.4189,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_011' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.2074,
  latency: 80,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 3378,
  confidence: 0.9816,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_012' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.7502,
  latency: 45,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 4923,
  confidence: 0.8772,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_013' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.4241,
  latency: 30,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 9509,
  confidence: 0.7052,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_014' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.2014,
  latency: 9,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 7826,
  confidence: 0.9664,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_015' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.6699,
  latency: 78,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 7744,
  confidence: 0.5812,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_016' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.0764,
  latency: 216,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 3704,
  confidence: 0.5902,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_017' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.3394,
  latency: 13,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 5324,
  confidence: 0.0175,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_018' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:CALIBRATES {
  strength: 0.1551,
  latency: 103,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 8670,
  confidence: 0.9607,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_019' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.9172,
  latency: 155,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 436,
  confidence: 0.0514,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_020' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.0329,
  latency: 57,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 7216,
  confidence: 0.735,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_021' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.2361,
  latency: 2,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 6119,
  confidence: 0.6965,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_022' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.2824,
  latency: 222,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 9056,
  confidence: 0.3032,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_023' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.3514,
  latency: 145,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 5601,
  confidence: 0.723,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_024' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.0811,
  latency: 208,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 3351,
  confidence: 0.0093,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_025' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.023,
  latency: 140,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 2836,
  confidence: 0.0357,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_026' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.3188,
  latency: 135,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 4396,
  confidence: 0.8494,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_027' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.8942,
  latency: 159,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 4282,
  confidence: 0.3718,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_028' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.3276,
  latency: 33,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7555,
  confidence: 0.8144,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_029' }),
      (b:Transformer { identifier: 'transformer_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.1437,
  latency: 229,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 2084,
  confidence: 0.2136,
  active: true
}]->(b);
