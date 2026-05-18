:param namespace => 'inferencecontext_02_02';
:param batchSize => 256;
:param threshold => 0.67;
:param maxDepth => 7;
:param timeoutSeconds => 55;
:param region => 'eu-west';
:param epoch => 15;
:param version => '4.7.3';

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_000' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.6278,
  latency: 177,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 2138,
  confidence: 0.0754,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_001' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.1665,
  latency: 179,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 5282,
  confidence: 0.6184,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_002' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.1791,
  latency: 243,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 9574,
  confidence: 0.7887,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_003' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.354,
  latency: 67,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 7156,
  confidence: 0.3537,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_004' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.8137,
  latency: 30,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 4492,
  confidence: 0.8745,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_005' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_006' })
MERGE (a)-[r_005:OBSERVES {
  strength: 0.7468,
  latency: 106,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 4542,
  confidence: 0.3403,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_006' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.0956,
  latency: 243,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 1885,
  confidence: 0.912,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_007' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.2342,
  latency: 25,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 1286,
  confidence: 0.975,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_008' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.1648,
  latency: 84,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 8574,
  confidence: 0.5596,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_009' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.9309,
  latency: 179,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 7264,
  confidence: 0.5698,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_010' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.7977,
  latency: 211,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 6127,
  confidence: 0.1894,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_011' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.9602,
  latency: 90,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6594,
  confidence: 0.8553,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_012' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.6065,
  latency: 182,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6618,
  confidence: 0.5115,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_013' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.7712,
  latency: 232,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 9749,
  confidence: 0.4729,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_014' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.851,
  latency: 14,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 5877,
  confidence: 0.343,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_015' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.0302,
  latency: 35,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 3403,
  confidence: 0.6764,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_016' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.1101,
  latency: 133,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 5616,
  confidence: 0.1647,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_017' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.307,
  latency: 165,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 4133,
  confidence: 0.6356,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_018' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.8065,
  latency: 217,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 5438,
  confidence: 0.682,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_019' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.1705,
  latency: 225,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 1969,
  confidence: 0.6497,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_020' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.1226,
  latency: 108,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 4608,
  confidence: 0.4207,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_021' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.9468,
  latency: 15,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 7378,
  confidence: 0.213,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_022' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.1942,
  latency: 178,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 3897,
  confidence: 0.8992,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_023' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.7259,
  latency: 54,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 1286,
  confidence: 0.1017,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_024' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.6814,
  latency: 222,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 1518,
  confidence: 0.0016,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_025' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.4879,
  latency: 32,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 7928,
  confidence: 0.5386,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_026' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.2008,
  latency: 187,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 8271,
  confidence: 0.1857,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_027' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.1053,
  latency: 9,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 4685,
  confidence: 0.4974,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_028' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.6338,
  latency: 134,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 7299,
  confidence: 0.3682,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_029' }),
      (b:InferenceContext { identifier: 'inferencecontext_03_config_managers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.7825,
  latency: 99,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 1484,
  confidence: 0.8556,
  active: true
}]->(b);
