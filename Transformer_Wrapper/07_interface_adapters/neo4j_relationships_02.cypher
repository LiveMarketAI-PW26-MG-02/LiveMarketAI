:param namespace => 'transformer_02_02';
:param batchSize => 64;
:param threshold => 0.74;
:param maxDepth => 12;
:param timeoutSeconds => 72;
:param region => 'us-east';
:param epoch => 95;
:param version => '1.8.0';

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_000' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.9964,
  latency: 196,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 5373,
  confidence: 0.5553,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_001' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.0657,
  latency: 219,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 9085,
  confidence: 0.0985,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_002' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.1893,
  latency: 111,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 8393,
  confidence: 0.7537,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_003' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_004' })
MERGE (a)-[r_003:OBSERVES {
  strength: 0.7275,
  latency: 9,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 3141,
  confidence: 0.7035,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_004' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.248,
  latency: 19,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 1936,
  confidence: 0.0531,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_005' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.728,
  latency: 28,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 1060,
  confidence: 0.6262,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_006' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.5826,
  latency: 170,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 312,
  confidence: 0.4989,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_007' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.0473,
  latency: 105,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 5377,
  confidence: 0.9542,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_008' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.8154,
  latency: 27,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 6047,
  confidence: 0.1436,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_009' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.1293,
  latency: 51,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 4466,
  confidence: 0.9954,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_010' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.6608,
  latency: 196,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 2005,
  confidence: 0.7816,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_011' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.5843,
  latency: 99,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1135,
  confidence: 0.2972,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_012' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.918,
  latency: 200,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 452,
  confidence: 0.383,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_013' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.3801,
  latency: 17,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 7544,
  confidence: 0.4742,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_014' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.7067,
  latency: 175,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 2390,
  confidence: 0.1728,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_015' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.9448,
  latency: 197,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 4763,
  confidence: 0.106,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_016' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.8068,
  latency: 53,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 3844,
  confidence: 0.1868,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_017' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.8897,
  latency: 152,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 4029,
  confidence: 0.1503,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_018' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.4256,
  latency: 27,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 6730,
  confidence: 0.5783,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_019' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_020' })
MERGE (a)-[r_019:CALIBRATES {
  strength: 0.9617,
  latency: 240,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 3546,
  confidence: 0.0252,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_020' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.8721,
  latency: 147,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 7756,
  confidence: 0.3712,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_021' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.2155,
  latency: 237,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 3396,
  confidence: 0.1985,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_022' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.6376,
  latency: 114,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 3145,
  confidence: 0.3005,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_023' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_024' })
MERGE (a)-[r_023:ROUTES_TO {
  strength: 0.9477,
  latency: 95,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 9049,
  confidence: 0.1056,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_024' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.6172,
  latency: 165,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 812,
  confidence: 0.4498,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_025' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.5844,
  latency: 107,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 5042,
  confidence: 0.1804,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_026' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.3355,
  latency: 232,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 1121,
  confidence: 0.5898,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_027' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.7338,
  latency: 86,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 9523,
  confidence: 0.4326,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_028' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.6232,
  latency: 250,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7728,
  confidence: 0.4785,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_07_interface_adapters_2_029' }),
      (b:Transformer { identifier: 'transformer_07_interface_adapters_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.8507,
  latency: 58,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 4995,
  confidence: 0.7225,
  active: true
}]->(b);
