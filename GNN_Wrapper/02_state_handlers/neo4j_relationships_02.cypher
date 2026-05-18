:param namespace => 'graphnetwork_02_02';
:param batchSize => 512;
:param threshold => 0.282;
:param maxDepth => 12;
:param timeoutSeconds => 98;
:param region => 'eu-west';
:param epoch => 31;
:param version => '5.8.7';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.4158,
  latency: 229,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 3086,
  confidence: 0.8885,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.2119,
  latency: 104,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 7447,
  confidence: 0.3024,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.6645,
  latency: 223,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6607,
  confidence: 0.9264,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.3743,
  latency: 90,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 4441,
  confidence: 0.0614,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.8743,
  latency: 149,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 5753,
  confidence: 0.2103,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.3138,
  latency: 143,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 4036,
  confidence: 0.9899,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.2396,
  latency: 239,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 9459,
  confidence: 0.51,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.7254,
  latency: 230,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 8281,
  confidence: 0.0789,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.9769,
  latency: 44,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 9089,
  confidence: 0.9064,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.9495,
  latency: 213,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 836,
  confidence: 0.2314,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.5453,
  latency: 138,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 8656,
  confidence: 0.692,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.4006,
  latency: 81,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 4680,
  confidence: 0.6209,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_013' })
MERGE (a)-[r_012:ROUTES_TO {
  strength: 0.2809,
  latency: 229,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 4950,
  confidence: 0.9879,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.8888,
  latency: 210,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 3591,
  confidence: 0.7733,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.9425,
  latency: 72,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 6536,
  confidence: 0.4614,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.2912,
  latency: 185,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 373,
  confidence: 0.8772,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.0272,
  latency: 210,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 4044,
  confidence: 0.8137,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_018' })
MERGE (a)-[r_017:PRODUCES {
  strength: 0.6775,
  latency: 191,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 3794,
  confidence: 0.4202,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.2224,
  latency: 238,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 2438,
  confidence: 0.4921,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.5557,
  latency: 136,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 7082,
  confidence: 0.0517,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.4053,
  latency: 99,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8876,
  confidence: 0.866,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.9957,
  latency: 248,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2119,
  confidence: 0.6539,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.1,
  latency: 99,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 3217,
  confidence: 0.7695,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.3884,
  latency: 200,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 8054,
  confidence: 0.8869,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.1071,
  latency: 110,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 7128,
  confidence: 0.1696,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.5383,
  latency: 236,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 2625,
  confidence: 0.4141,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.8746,
  latency: 133,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 599,
  confidence: 0.0316,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.0821,
  latency: 127,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 594,
  confidence: 0.7256,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.7263,
  latency: 64,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 3525,
  confidence: 0.1945,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_02_state_handlers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.8557,
  latency: 88,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 5230,
  confidence: 0.4631,
  active: true
}]->(b);
