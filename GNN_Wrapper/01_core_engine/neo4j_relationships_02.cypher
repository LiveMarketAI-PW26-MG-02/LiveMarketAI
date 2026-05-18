:param namespace => 'graphnetwork_02_02';
:param batchSize => 128;
:param threshold => 0.283;
:param maxDepth => 8;
:param timeoutSeconds => 30;
:param region => 'eu-west';
:param epoch => 4;
:param version => '2.8.1';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.3595,
  latency: 132,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 5301,
  confidence: 0.5159,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.1952,
  latency: 95,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 5815,
  confidence: 0.2078,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.6453,
  latency: 162,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 3220,
  confidence: 0.8306,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.9451,
  latency: 88,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 4212,
  confidence: 0.7711,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.9995,
  latency: 223,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 4321,
  confidence: 0.9053,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.0372,
  latency: 139,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 9617,
  confidence: 0.1578,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.1223,
  latency: 27,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 7280,
  confidence: 0.3237,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.1741,
  latency: 47,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 1852,
  confidence: 0.6118,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.4353,
  latency: 4,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 6380,
  confidence: 0.062,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.4359,
  latency: 107,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 9401,
  confidence: 0.7316,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.6918,
  latency: 163,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 9258,
  confidence: 0.1218,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_012' })
MERGE (a)-[r_011:VALIDATES {
  strength: 0.7607,
  latency: 37,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 2706,
  confidence: 0.4056,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.7655,
  latency: 223,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8097,
  confidence: 0.0511,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.7968,
  latency: 188,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 4008,
  confidence: 0.7489,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.3548,
  latency: 152,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 6476,
  confidence: 0.7046,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.7615,
  latency: 162,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 9729,
  confidence: 0.3665,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.1556,
  latency: 58,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5687,
  confidence: 0.4339,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.9793,
  latency: 60,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 5330,
  confidence: 0.1397,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.8704,
  latency: 13,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 7156,
  confidence: 0.3729,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_020' })
MERGE (a)-[r_019:PRODUCES {
  strength: 0.3611,
  latency: 140,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 9054,
  confidence: 0.2582,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.0056,
  latency: 52,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 4042,
  confidence: 0.6948,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.1833,
  latency: 61,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 9270,
  confidence: 0.3497,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.5002,
  latency: 67,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 2569,
  confidence: 0.0078,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.1394,
  latency: 171,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 4991,
  confidence: 0.7729,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.9484,
  latency: 157,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 1379,
  confidence: 0.5076,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.894,
  latency: 46,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 8265,
  confidence: 0.5426,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.4592,
  latency: 241,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 2810,
  confidence: 0.1655,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.984,
  latency: 105,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 5478,
  confidence: 0.9312,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.3567,
  latency: 127,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 729,
  confidence: 0.5201,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_01_core_engine_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.6095,
  latency: 170,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 7722,
  confidence: 0.0337,
  active: true
}]->(b);
