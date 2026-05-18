:param namespace => 'batchinference_02_02';
:param batchSize => 128;
:param threshold => 0.712;
:param maxDepth => 10;
:param timeoutSeconds => 34;
:param region => 'ap-south';
:param epoch => 21;
:param version => '5.5.3';

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_000' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.3236,
  latency: 148,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 4256,
  confidence: 0.1223,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_001' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.8466,
  latency: 142,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 2472,
  confidence: 0.0479,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_002' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.4029,
  latency: 149,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 9284,
  confidence: 0.593,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_003' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:OBSERVES {
  strength: 0.0869,
  latency: 13,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 9120,
  confidence: 0.4635,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_004' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.9324,
  latency: 83,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 2486,
  confidence: 0.6113,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_005' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.5782,
  latency: 133,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 8859,
  confidence: 0.2214,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_006' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.7451,
  latency: 155,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 2867,
  confidence: 0.2644,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_007' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.8541,
  latency: 144,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 6559,
  confidence: 0.6436,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_008' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.4876,
  latency: 207,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 2408,
  confidence: 0.4091,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_009' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.4439,
  latency: 227,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 5792,
  confidence: 0.5183,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_010' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.8019,
  latency: 131,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 4552,
  confidence: 0.8054,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_011' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.8993,
  latency: 139,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 8821,
  confidence: 0.579,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_012' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.7491,
  latency: 208,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 1440,
  confidence: 0.8255,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_013' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.3514,
  latency: 135,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 4556,
  confidence: 0.8914,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_014' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.3813,
  latency: 22,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 2509,
  confidence: 0.3695,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_015' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.3522,
  latency: 77,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 9628,
  confidence: 0.977,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_016' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.5294,
  latency: 145,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 7397,
  confidence: 0.4279,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_017' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.1089,
  latency: 203,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 845,
  confidence: 0.0718,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_018' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.3989,
  latency: 228,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 9472,
  confidence: 0.8101,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_019' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.3179,
  latency: 192,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 1436,
  confidence: 0.4776,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_020' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.301,
  latency: 52,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 596,
  confidence: 0.6455,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_021' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.2454,
  latency: 243,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 6018,
  confidence: 0.2113,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_022' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.0501,
  latency: 49,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 953,
  confidence: 0.0977,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_023' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.2587,
  latency: 230,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 5269,
  confidence: 0.4587,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_024' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.6521,
  latency: 201,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 1039,
  confidence: 0.18,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_025' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.5858,
  latency: 12,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 8905,
  confidence: 0.7514,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_026' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.5597,
  latency: 231,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 4057,
  confidence: 0.3765,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_027' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.5363,
  latency: 157,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 4023,
  confidence: 0.5573,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_028' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.4809,
  latency: 11,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 3738,
  confidence: 0.3217,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_029' }),
      (b:BatchInference { identifier: 'batchinference_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.7896,
  latency: 123,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 829,
  confidence: 0.4608,
  active: true
}]->(b);
