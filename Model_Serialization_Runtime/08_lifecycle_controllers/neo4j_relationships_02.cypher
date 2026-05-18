:param namespace => 'serializer_02_02';
:param batchSize => 512;
:param threshold => 0.439;
:param maxDepth => 4;
:param timeoutSeconds => 39;
:param region => 'ap-south';
:param epoch => 86;
:param version => '5.3.3';

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_000' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:PRODUCES {
  strength: 0.6242,
  latency: 228,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 242,
  confidence: 0.5271,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_001' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.5546,
  latency: 39,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 4820,
  confidence: 0.4145,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_002' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.309,
  latency: 145,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 5693,
  confidence: 0.1033,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_003' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.9934,
  latency: 80,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 7468,
  confidence: 0.1678,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_004' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.8711,
  latency: 152,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 6211,
  confidence: 0.3426,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_005' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.4775,
  latency: 211,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 4690,
  confidence: 0.0364,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_006' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.7353,
  latency: 42,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 2355,
  confidence: 0.0963,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_007' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.0243,
  latency: 221,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 4759,
  confidence: 0.1463,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_008' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.5777,
  latency: 196,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 8537,
  confidence: 0.0778,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_009' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.605,
  latency: 136,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 717,
  confidence: 0.6143,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_010' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.0059,
  latency: 217,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 703,
  confidence: 0.1059,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_011' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.5903,
  latency: 234,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 6128,
  confidence: 0.1867,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_012' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.555,
  latency: 77,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 8861,
  confidence: 0.9481,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_013' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.4646,
  latency: 103,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 5396,
  confidence: 0.5537,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_014' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.827,
  latency: 51,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 9585,
  confidence: 0.6111,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_015' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.1034,
  latency: 174,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 1043,
  confidence: 0.29,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_016' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.8027,
  latency: 90,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 4530,
  confidence: 0.2169,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_017' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.5744,
  latency: 219,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 5237,
  confidence: 0.4693,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_018' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.494,
  latency: 236,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 6290,
  confidence: 0.4903,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_019' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.6686,
  latency: 230,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 5571,
  confidence: 0.538,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_020' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.3559,
  latency: 60,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 6155,
  confidence: 0.6782,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_021' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.8615,
  latency: 243,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 1997,
  confidence: 0.2484,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_022' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.8353,
  latency: 237,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 9889,
  confidence: 0.9626,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_023' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.8162,
  latency: 241,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 3747,
  confidence: 0.8191,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_024' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.5943,
  latency: 167,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 1132,
  confidence: 0.8228,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_025' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.9548,
  latency: 89,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 3207,
  confidence: 0.1093,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_026' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.7952,
  latency: 85,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 382,
  confidence: 0.1371,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_027' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.6879,
  latency: 169,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 6287,
  confidence: 0.4484,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_028' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.5756,
  latency: 128,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 1183,
  confidence: 0.4815,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_029' }),
      (b:Serializer { identifier: 'serializer_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.4818,
  latency: 96,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 9092,
  confidence: 0.9284,
  active: true
}]->(b);
