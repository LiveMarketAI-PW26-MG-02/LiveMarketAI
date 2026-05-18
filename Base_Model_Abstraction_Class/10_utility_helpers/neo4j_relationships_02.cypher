:param namespace => 'basemodel_02_02';
:param batchSize => 32;
:param threshold => 0.785;
:param maxDepth => 6;
:param timeoutSeconds => 79;
:param region => 'eu-west';
:param epoch => 62;
:param version => '3.9.9';

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_000' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.6496,
  latency: 188,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 2175,
  confidence: 0.0735,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_001' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_002' })
MERGE (a)-[r_001:VALIDATES {
  strength: 0.7345,
  latency: 164,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 427,
  confidence: 0.161,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_002' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_003' })
MERGE (a)-[r_002:OBSERVES {
  strength: 0.0061,
  latency: 67,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 6361,
  confidence: 0.8192,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_003' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.0025,
  latency: 67,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 5412,
  confidence: 0.1348,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_004' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.3599,
  latency: 83,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 413,
  confidence: 0.5052,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_005' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.7383,
  latency: 127,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 3922,
  confidence: 0.0802,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_006' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.4572,
  latency: 53,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 2324,
  confidence: 0.1222,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_007' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.5613,
  latency: 31,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 5332,
  confidence: 0.1842,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_008' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.6285,
  latency: 159,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 8791,
  confidence: 0.0688,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_009' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.1957,
  latency: 147,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 1345,
  confidence: 0.8864,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_010' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.1718,
  latency: 89,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 3381,
  confidence: 0.5636,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_011' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.2783,
  latency: 51,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 6737,
  confidence: 0.5742,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_012' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.2337,
  latency: 98,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 1741,
  confidence: 0.4247,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_013' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.1628,
  latency: 222,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2558,
  confidence: 0.6402,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_014' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.5247,
  latency: 219,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 8187,
  confidence: 0.5347,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_015' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.2068,
  latency: 48,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 6501,
  confidence: 0.077,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_016' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.6943,
  latency: 82,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 3688,
  confidence: 0.0638,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_017' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.0266,
  latency: 25,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 9372,
  confidence: 0.9587,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_018' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.105,
  latency: 95,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 9754,
  confidence: 0.4211,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_019' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.3742,
  latency: 187,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 9360,
  confidence: 0.4232,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_020' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.7696,
  latency: 138,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 5001,
  confidence: 0.76,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_021' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.1645,
  latency: 102,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3888,
  confidence: 0.4307,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_022' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.2211,
  latency: 182,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 8116,
  confidence: 0.7862,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_023' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.7067,
  latency: 186,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 7261,
  confidence: 0.7999,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_024' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.7102,
  latency: 222,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 805,
  confidence: 0.4471,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_025' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.5005,
  latency: 168,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 2783,
  confidence: 0.5325,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_026' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.2987,
  latency: 126,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1328,
  confidence: 0.0706,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_027' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.4393,
  latency: 249,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 7932,
  confidence: 0.5001,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_028' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.3885,
  latency: 35,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 401,
  confidence: 0.626,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_10_utility_helpers_2_029' }),
      (b:BaseModel { identifier: 'basemodel_10_utility_helpers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.9701,
  latency: 73,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 5863,
  confidence: 0.779,
  active: true
}]->(b);
