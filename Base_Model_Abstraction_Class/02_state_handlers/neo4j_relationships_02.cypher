:param namespace => 'basemodel_02_02';
:param batchSize => 64;
:param threshold => 0.29;
:param maxDepth => 3;
:param timeoutSeconds => 109;
:param region => 'us-east';
:param epoch => 2;
:param version => '5.8.3';

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_000' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.4132,
  latency: 133,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 8405,
  confidence: 0.6476,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_001' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.8134,
  latency: 45,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 5168,
  confidence: 0.0638,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_002' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.9941,
  latency: 186,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 8921,
  confidence: 0.0063,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_003' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.7452,
  latency: 120,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 7513,
  confidence: 0.1754,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_004' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.2614,
  latency: 165,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 2119,
  confidence: 0.3355,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_005' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.7117,
  latency: 69,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 7244,
  confidence: 0.6857,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_006' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.2956,
  latency: 238,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 1499,
  confidence: 0.88,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_007' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.1698,
  latency: 232,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 3422,
  confidence: 0.9447,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_008' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.1919,
  latency: 100,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 9950,
  confidence: 0.2392,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_009' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.4721,
  latency: 136,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 534,
  confidence: 0.4372,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_010' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.5703,
  latency: 79,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 6515,
  confidence: 0.6226,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_011' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.5652,
  latency: 44,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 639,
  confidence: 0.0269,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_012' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.622,
  latency: 42,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 2423,
  confidence: 0.7007,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_013' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.0416,
  latency: 178,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 1211,
  confidence: 0.7368,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_014' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.8565,
  latency: 196,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 3365,
  confidence: 0.8176,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_015' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.8797,
  latency: 194,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 1854,
  confidence: 0.2466,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_016' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.112,
  latency: 9,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 4808,
  confidence: 0.4771,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_017' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.0979,
  latency: 194,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 4924,
  confidence: 0.3191,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_018' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.2612,
  latency: 90,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4730,
  confidence: 0.0484,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_019' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.9103,
  latency: 197,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 8353,
  confidence: 0.4761,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_020' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.6183,
  latency: 8,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 611,
  confidence: 0.4364,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_021' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.3468,
  latency: 181,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 8912,
  confidence: 0.5661,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_022' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.5745,
  latency: 74,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 7244,
  confidence: 0.0013,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_023' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.2883,
  latency: 193,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 171,
  confidence: 0.3478,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_024' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.4915,
  latency: 204,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 8203,
  confidence: 0.5926,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_025' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.578,
  latency: 41,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 3617,
  confidence: 0.9383,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_026' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.4983,
  latency: 29,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 8132,
  confidence: 0.7879,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_027' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.6279,
  latency: 92,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6674,
  confidence: 0.9285,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_028' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.4221,
  latency: 166,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 6194,
  confidence: 0.2061,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_02_state_handlers_2_029' }),
      (b:BaseModel { identifier: 'basemodel_02_state_handlers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.4281,
  latency: 140,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 2903,
  confidence: 0.3793,
  active: true
}]->(b);
