:param namespace => 'serializer_02_02';
:param batchSize => 256;
:param threshold => 0.149;
:param maxDepth => 9;
:param timeoutSeconds => 107;
:param region => 'us-west';
:param epoch => 8;
:param version => '2.4.4';

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_000' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.3179,
  latency: 187,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 5431,
  confidence: 0.3845,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_001' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.5987,
  latency: 189,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 5980,
  confidence: 0.8862,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_002' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.4423,
  latency: 138,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 6689,
  confidence: 0.9439,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_003' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.3185,
  latency: 92,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 7218,
  confidence: 0.1669,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_004' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.0773,
  latency: 125,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 3618,
  confidence: 0.7993,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_005' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.3269,
  latency: 56,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 4229,
  confidence: 0.464,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_006' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.5333,
  latency: 27,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 7416,
  confidence: 0.1885,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_007' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.8626,
  latency: 24,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 5982,
  confidence: 0.5352,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_008' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.1064,
  latency: 226,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 484,
  confidence: 0.3815,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_009' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.4582,
  latency: 190,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 1258,
  confidence: 0.8763,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_010' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.838,
  latency: 95,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 4689,
  confidence: 0.3108,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_011' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_012' })
MERGE (a)-[r_011:VALIDATES {
  strength: 0.1072,
  latency: 49,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 4320,
  confidence: 0.6014,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_012' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.9143,
  latency: 186,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 5533,
  confidence: 0.8861,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_013' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.7554,
  latency: 201,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 8506,
  confidence: 0.8403,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_014' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.2624,
  latency: 40,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 7161,
  confidence: 0.0644,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_015' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.1159,
  latency: 142,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 6665,
  confidence: 0.8675,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_016' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.6972,
  latency: 157,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 4617,
  confidence: 0.3006,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_017' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.3574,
  latency: 124,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 3023,
  confidence: 0.3302,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_018' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.5078,
  latency: 39,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6100,
  confidence: 0.4879,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_019' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.7805,
  latency: 167,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 2643,
  confidence: 0.0879,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_020' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.5121,
  latency: 112,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 6376,
  confidence: 0.205,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_021' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.3386,
  latency: 192,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 4457,
  confidence: 0.3295,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_022' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.1756,
  latency: 58,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 6596,
  confidence: 0.5426,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_023' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.552,
  latency: 39,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6532,
  confidence: 0.1792,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_024' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.0225,
  latency: 87,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 786,
  confidence: 0.9164,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_025' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.0242,
  latency: 217,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 7368,
  confidence: 0.9628,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_026' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.8842,
  latency: 242,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 5598,
  confidence: 0.8118,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_027' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.9751,
  latency: 1,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 8889,
  confidence: 0.2781,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_028' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.5786,
  latency: 207,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 2448,
  confidence: 0.2908,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_10_utility_helpers_2_029' }),
      (b:Serializer { identifier: 'serializer_10_utility_helpers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.9932,
  latency: 111,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 4038,
  confidence: 0.2003,
  active: true
}]->(b);
