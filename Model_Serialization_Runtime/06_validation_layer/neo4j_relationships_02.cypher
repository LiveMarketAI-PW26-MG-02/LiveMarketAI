:param namespace => 'serializer_02_02';
:param batchSize => 32;
:param threshold => 0.594;
:param maxDepth => 6;
:param timeoutSeconds => 47;
:param region => 'us-west';
:param epoch => 14;
:param version => '2.7.8';

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_000' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.1644,
  latency: 209,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 6226,
  confidence: 0.9623,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_001' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.6219,
  latency: 95,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 2717,
  confidence: 0.1412,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_002' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.4544,
  latency: 178,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 1729,
  confidence: 0.7324,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_003' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.3791,
  latency: 140,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 9352,
  confidence: 0.851,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_004' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.525,
  latency: 188,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 7660,
  confidence: 0.5189,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_005' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.505,
  latency: 183,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2668,
  confidence: 0.9303,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_006' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.1429,
  latency: 247,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 720,
  confidence: 0.7274,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_007' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.9759,
  latency: 203,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 9603,
  confidence: 0.8166,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_008' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.9955,
  latency: 181,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 5110,
  confidence: 0.9925,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_009' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.5995,
  latency: 230,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 3016,
  confidence: 0.2252,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_010' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.2669,
  latency: 150,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 7177,
  confidence: 0.3867,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_011' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.0262,
  latency: 97,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 8239,
  confidence: 0.0732,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_012' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_013' })
MERGE (a)-[r_012:ROUTES_TO {
  strength: 0.0825,
  latency: 139,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 3507,
  confidence: 0.3988,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_013' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.1966,
  latency: 24,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 692,
  confidence: 0.8495,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_014' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.2074,
  latency: 146,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 5437,
  confidence: 0.9553,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_015' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.8829,
  latency: 44,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 239,
  confidence: 0.2946,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_016' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.5016,
  latency: 188,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 5407,
  confidence: 0.1259,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_017' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.7717,
  latency: 29,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 4725,
  confidence: 0.9954,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_018' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.6771,
  latency: 202,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6025,
  confidence: 0.7924,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_019' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.199,
  latency: 188,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 4839,
  confidence: 0.9708,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_020' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.4961,
  latency: 218,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 5452,
  confidence: 0.1458,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_021' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.2804,
  latency: 10,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 4024,
  confidence: 0.4702,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_022' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.6763,
  latency: 176,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 6558,
  confidence: 0.3792,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_023' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.8623,
  latency: 35,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6225,
  confidence: 0.7375,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_024' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.6357,
  latency: 240,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 211,
  confidence: 0.4554,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_025' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.3558,
  latency: 210,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 6068,
  confidence: 0.3164,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_026' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.294,
  latency: 142,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6057,
  confidence: 0.0559,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_027' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.4085,
  latency: 166,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 9810,
  confidence: 0.3508,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_028' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.7072,
  latency: 19,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 2848,
  confidence: 0.7345,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_06_validation_layer_2_029' }),
      (b:Serializer { identifier: 'serializer_06_validation_layer_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.6773,
  latency: 77,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 5007,
  confidence: 0.1136,
  active: true
}]->(b);
