:param namespace => 'graphnetwork_02_02';
:param batchSize => 64;
:param threshold => 0.85;
:param maxDepth => 8;
:param timeoutSeconds => 85;
:param region => 'eu-west';
:param epoch => 32;
:param version => '2.9.6';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.2347,
  latency: 135,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 5706,
  confidence: 0.492,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.7044,
  latency: 233,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1229,
  confidence: 0.7492,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.7609,
  latency: 44,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 8395,
  confidence: 0.213,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.0988,
  latency: 223,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 4444,
  confidence: 0.8168,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.1591,
  latency: 7,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7283,
  confidence: 0.2259,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.2369,
  latency: 145,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 5471,
  confidence: 0.5528,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.9498,
  latency: 168,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 1466,
  confidence: 0.743,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.4736,
  latency: 4,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4859,
  confidence: 0.0686,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.5574,
  latency: 61,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 1302,
  confidence: 0.536,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.3065,
  latency: 42,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 3286,
  confidence: 0.5367,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.641,
  latency: 18,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 8592,
  confidence: 0.8404,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.4093,
  latency: 163,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 116,
  confidence: 0.8985,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_013' })
MERGE (a)-[r_012:ROUTES_TO {
  strength: 0.1083,
  latency: 7,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 3856,
  confidence: 0.3209,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_014' })
MERGE (a)-[r_013:MONITORS {
  strength: 0.8885,
  latency: 182,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 6655,
  confidence: 0.1572,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.4593,
  latency: 89,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 6602,
  confidence: 0.0467,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.5544,
  latency: 113,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 9777,
  confidence: 0.2732,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.4742,
  latency: 170,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 3865,
  confidence: 0.1915,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_018' })
MERGE (a)-[r_017:PRODUCES {
  strength: 0.9334,
  latency: 178,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 9496,
  confidence: 0.1675,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.4015,
  latency: 49,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 586,
  confidence: 0.6929,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.9002,
  latency: 16,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 1805,
  confidence: 0.125,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.5457,
  latency: 239,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 5889,
  confidence: 0.4322,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.3016,
  latency: 119,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3708,
  confidence: 0.3469,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.9317,
  latency: 148,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 7634,
  confidence: 0.2993,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.6139,
  latency: 158,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 2628,
  confidence: 0.558,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.9643,
  latency: 152,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 2649,
  confidence: 0.5707,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.5701,
  latency: 184,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 7482,
  confidence: 0.2077,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.377,
  latency: 104,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 7353,
  confidence: 0.8098,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.9551,
  latency: 26,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 6966,
  confidence: 0.4564,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.252,
  latency: 92,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 6731,
  confidence: 0.6185,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_10_utility_helpers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.264,
  latency: 42,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 537,
  confidence: 0.0769,
  active: true
}]->(b);
