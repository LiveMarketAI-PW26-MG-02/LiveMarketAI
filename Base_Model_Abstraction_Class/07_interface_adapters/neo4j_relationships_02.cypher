:param namespace => 'basemodel_02_02';
:param batchSize => 32;
:param threshold => 0.272;
:param maxDepth => 12;
:param timeoutSeconds => 70;
:param region => 'us-east';
:param epoch => 94;
:param version => '3.5.9';

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_000' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.9568,
  latency: 166,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 3146,
  confidence: 0.8383,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_001' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.3363,
  latency: 42,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 7864,
  confidence: 0.8576,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_002' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.2502,
  latency: 234,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 3723,
  confidence: 0.1611,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_003' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.9711,
  latency: 198,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6377,
  confidence: 0.533,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_004' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.2122,
  latency: 107,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 5224,
  confidence: 0.682,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_005' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.232,
  latency: 119,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 8783,
  confidence: 0.962,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_006' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.1605,
  latency: 176,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 9178,
  confidence: 0.3183,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_007' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.9149,
  latency: 231,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 7793,
  confidence: 0.4932,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_008' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.5632,
  latency: 26,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 8250,
  confidence: 0.7619,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_009' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.1621,
  latency: 227,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6124,
  confidence: 0.3797,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_010' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.9817,
  latency: 36,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 9640,
  confidence: 0.2826,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_011' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.385,
  latency: 141,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 5242,
  confidence: 0.7705,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_012' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.2046,
  latency: 32,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 7558,
  confidence: 0.6295,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_013' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.4807,
  latency: 238,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 9000,
  confidence: 0.9581,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_014' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.3603,
  latency: 155,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 5019,
  confidence: 0.2931,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_015' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.7093,
  latency: 151,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6989,
  confidence: 0.0098,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_016' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_017' })
MERGE (a)-[r_016:PRODUCES {
  strength: 0.2058,
  latency: 130,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 3987,
  confidence: 0.6691,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_017' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.9267,
  latency: 50,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 129,
  confidence: 0.2666,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_018' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.0876,
  latency: 72,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 9413,
  confidence: 0.6931,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_019' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.35,
  latency: 182,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 8828,
  confidence: 0.8243,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_020' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.5731,
  latency: 46,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 1765,
  confidence: 0.2106,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_021' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.2675,
  latency: 226,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 5400,
  confidence: 0.6748,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_022' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.4051,
  latency: 179,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 1202,
  confidence: 0.5966,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_023' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.1105,
  latency: 192,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 8527,
  confidence: 0.1479,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_024' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.8708,
  latency: 6,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 992,
  confidence: 0.9936,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_025' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.1611,
  latency: 186,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 9131,
  confidence: 0.1334,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_026' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.2551,
  latency: 37,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 2691,
  confidence: 0.1517,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_027' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.5885,
  latency: 206,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 2722,
  confidence: 0.3093,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_028' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.5605,
  latency: 106,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 9006,
  confidence: 0.75,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_07_interface_adapters_2_029' }),
      (b:BaseModel { identifier: 'basemodel_07_interface_adapters_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.2362,
  latency: 36,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 194,
  confidence: 0.2419,
  active: true
}]->(b);
