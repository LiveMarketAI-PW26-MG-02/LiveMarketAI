:param namespace => 'multimodal_02_02';
:param batchSize => 256;
:param threshold => 0.704;
:param maxDepth => 4;
:param timeoutSeconds => 113;
:param region => 'ap-south';
:param epoch => 18;
:param version => '5.0.4';

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_000' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.2387,
  latency: 65,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 9510,
  confidence: 0.0693,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_001' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.5768,
  latency: 50,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 4546,
  confidence: 0.5669,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_002' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.6629,
  latency: 51,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 962,
  confidence: 0.2944,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_003' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.9295,
  latency: 87,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5300,
  confidence: 0.034,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_004' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.5938,
  latency: 15,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 1665,
  confidence: 0.3309,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_005' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.2141,
  latency: 65,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 5225,
  confidence: 0.6646,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_006' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.8594,
  latency: 246,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 3153,
  confidence: 0.4167,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_007' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.5029,
  latency: 191,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 5913,
  confidence: 0.6155,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_008' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.5417,
  latency: 222,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9701,
  confidence: 0.6288,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_009' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.6368,
  latency: 242,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 3353,
  confidence: 0.8937,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_010' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.7918,
  latency: 193,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 5253,
  confidence: 0.9072,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_011' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.965,
  latency: 225,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 5290,
  confidence: 0.3025,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_012' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.1393,
  latency: 234,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 1495,
  confidence: 0.2457,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_013' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_014' })
MERGE (a)-[r_013:MONITORS {
  strength: 0.5115,
  latency: 45,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 7893,
  confidence: 0.3198,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_014' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.6456,
  latency: 9,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 2268,
  confidence: 0.6675,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_015' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.6102,
  latency: 23,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 2171,
  confidence: 0.3991,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_016' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_017' })
MERGE (a)-[r_016:PRODUCES {
  strength: 0.8123,
  latency: 131,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 4866,
  confidence: 0.4313,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_017' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.5273,
  latency: 201,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 5298,
  confidence: 0.6496,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_018' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.6621,
  latency: 49,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 409,
  confidence: 0.7426,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_019' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.2132,
  latency: 219,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 173,
  confidence: 0.5619,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_020' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.8621,
  latency: 215,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 7374,
  confidence: 0.174,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_021' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.2263,
  latency: 39,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2381,
  confidence: 0.1474,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_022' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.8281,
  latency: 203,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 3166,
  confidence: 0.9828,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_023' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.4684,
  latency: 75,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 867,
  confidence: 0.5145,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_024' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.2735,
  latency: 131,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 5254,
  confidence: 0.2115,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_025' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.965,
  latency: 55,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 5709,
  confidence: 0.0133,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_026' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.1652,
  latency: 168,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 2756,
  confidence: 0.5663,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_027' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.7189,
  latency: 136,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 8034,
  confidence: 0.7393,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_028' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.8124,
  latency: 71,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 1609,
  confidence: 0.8027,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_02_state_handlers_2_029' }),
      (b:Multimodal { identifier: 'multimodal_02_state_handlers_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.8319,
  latency: 30,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 4004,
  confidence: 0.8745,
  active: true
}]->(b);
