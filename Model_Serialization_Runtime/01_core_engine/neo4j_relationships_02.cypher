:param namespace => 'serializer_02_02';
:param batchSize => 32;
:param threshold => 0.661;
:param maxDepth => 11;
:param timeoutSeconds => 98;
:param region => 'eu-west';
:param epoch => 77;
:param version => '2.6.8';

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_000' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.224,
  latency: 3,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 6735,
  confidence: 0.5725,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_001' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.9152,
  latency: 228,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 3740,
  confidence: 0.117,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_002' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.3915,
  latency: 211,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 2147,
  confidence: 0.3598,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_003' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.8839,
  latency: 220,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 762,
  confidence: 0.4591,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_004' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.9215,
  latency: 219,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 7181,
  confidence: 0.1761,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_005' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.4202,
  latency: 53,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 9567,
  confidence: 0.6382,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_006' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.3441,
  latency: 241,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 5940,
  confidence: 0.5372,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_007' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.682,
  latency: 226,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 708,
  confidence: 0.1918,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_008' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.9703,
  latency: 225,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 273,
  confidence: 0.3308,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_009' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_010' })
MERGE (a)-[r_009:PRODUCES {
  strength: 0.8035,
  latency: 203,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 2588,
  confidence: 0.713,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_010' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.3858,
  latency: 218,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9295,
  confidence: 0.4913,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_011' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.004,
  latency: 51,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 7334,
  confidence: 0.1846,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_012' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.7312,
  latency: 65,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 2502,
  confidence: 0.907,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_013' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.4051,
  latency: 111,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 3542,
  confidence: 0.2615,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_014' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_015' })
MERGE (a)-[r_014:TRIGGERS {
  strength: 0.5534,
  latency: 72,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 3818,
  confidence: 0.2636,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_015' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.1337,
  latency: 152,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 3242,
  confidence: 0.732,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_016' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.5937,
  latency: 140,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 5043,
  confidence: 0.0802,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_017' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.7446,
  latency: 159,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 3757,
  confidence: 0.1918,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_018' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_019' })
MERGE (a)-[r_018:CALIBRATES {
  strength: 0.7776,
  latency: 179,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6095,
  confidence: 0.8576,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_019' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.7193,
  latency: 218,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 636,
  confidence: 0.7641,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_020' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.8,
  latency: 186,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 778,
  confidence: 0.052,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_021' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.8364,
  latency: 93,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 9318,
  confidence: 0.7171,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_022' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.234,
  latency: 76,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 1548,
  confidence: 0.126,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_023' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_024' })
MERGE (a)-[r_023:ROUTES_TO {
  strength: 0.3176,
  latency: 27,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 2526,
  confidence: 0.6232,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_024' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.0369,
  latency: 236,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 6459,
  confidence: 0.6958,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_025' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.4916,
  latency: 121,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 6085,
  confidence: 0.8959,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_026' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.3107,
  latency: 122,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 3452,
  confidence: 0.8874,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_027' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.6753,
  latency: 21,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 3236,
  confidence: 0.5718,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_028' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.1865,
  latency: 213,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 8874,
  confidence: 0.6198,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_01_core_engine_2_029' }),
      (b:Serializer { identifier: 'serializer_01_core_engine_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.5685,
  latency: 122,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 6314,
  confidence: 0.9863,
  active: true
}]->(b);
