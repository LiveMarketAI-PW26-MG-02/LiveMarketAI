:param namespace => 'multimodal_02_02';
:param batchSize => 256;
:param threshold => 0.211;
:param maxDepth => 7;
:param timeoutSeconds => 86;
:param region => 'us-west';
:param epoch => 29;
:param version => '3.2.8';

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_000' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.0922,
  latency: 119,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 9720,
  confidence: 0.5309,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_001' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.1025,
  latency: 177,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 2197,
  confidence: 0.551,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_002' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.1669,
  latency: 99,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 3667,
  confidence: 0.0322,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_003' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.6835,
  latency: 73,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 4001,
  confidence: 0.0048,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_004' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.7038,
  latency: 205,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 9762,
  confidence: 0.8321,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_005' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.5283,
  latency: 219,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 6282,
  confidence: 0.3034,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_006' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.0897,
  latency: 220,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 8294,
  confidence: 0.792,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_007' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.9284,
  latency: 160,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 1618,
  confidence: 0.9131,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_008' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.2439,
  latency: 45,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 9014,
  confidence: 0.8989,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_009' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.5867,
  latency: 83,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 3462,
  confidence: 0.8183,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_010' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.8722,
  latency: 161,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 7157,
  confidence: 0.4896,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_011' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.0987,
  latency: 246,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 7095,
  confidence: 0.0522,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_012' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.7984,
  latency: 100,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 6325,
  confidence: 0.6482,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_013' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.2231,
  latency: 97,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 8495,
  confidence: 0.4405,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_014' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.1162,
  latency: 236,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 7558,
  confidence: 0.3599,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_015' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.6917,
  latency: 65,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 5963,
  confidence: 0.8612,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_016' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.6825,
  latency: 174,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 201,
  confidence: 0.2696,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_017' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.2633,
  latency: 128,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 7066,
  confidence: 0.5525,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_018' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.6438,
  latency: 193,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 3232,
  confidence: 0.4501,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_019' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.5223,
  latency: 8,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5194,
  confidence: 0.4584,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_020' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.6908,
  latency: 25,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 4252,
  confidence: 0.4519,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_021' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.2265,
  latency: 12,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 3441,
  confidence: 0.9569,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_022' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.2874,
  latency: 197,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 9388,
  confidence: 0.1391,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_023' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.2809,
  latency: 74,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 5733,
  confidence: 0.9106,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_024' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.329,
  latency: 16,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 2291,
  confidence: 0.4363,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_025' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.3205,
  latency: 110,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 9945,
  confidence: 0.8421,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_026' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.1884,
  latency: 122,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 7876,
  confidence: 0.8917,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_027' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_028' })
MERGE (a)-[r_027:CALIBRATES {
  strength: 0.3139,
  latency: 56,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 4574,
  confidence: 0.4398,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_028' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.1231,
  latency: 172,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 987,
  confidence: 0.2067,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_10_utility_helpers_2_029' }),
      (b:Multimodal { identifier: 'multimodal_10_utility_helpers_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.1397,
  latency: 52,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 4534,
  confidence: 0.4553,
  active: true
}]->(b);
