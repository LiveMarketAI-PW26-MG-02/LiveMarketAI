:param namespace => 'explainability_02_02';
:param batchSize => 256;
:param threshold => 0.57;
:param maxDepth => 3;
:param timeoutSeconds => 84;
:param region => 'us-west';
:param epoch => 25;
:param version => '1.6.4';

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_000' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.095,
  latency: 203,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 644,
  confidence: 0.8229,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_001' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.2649,
  latency: 248,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 4996,
  confidence: 0.8065,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_002' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.2794,
  latency: 107,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 3070,
  confidence: 0.2782,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_003' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.3995,
  latency: 77,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 342,
  confidence: 0.8985,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_004' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.5383,
  latency: 122,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 610,
  confidence: 0.363,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_005' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.9544,
  latency: 174,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 1651,
  confidence: 0.4338,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_006' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.366,
  latency: 122,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 670,
  confidence: 0.5436,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_007' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.5526,
  latency: 75,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6676,
  confidence: 0.6474,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_008' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.3283,
  latency: 70,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 1115,
  confidence: 0.7685,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_009' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.7807,
  latency: 6,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 3674,
  confidence: 0.5749,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_010' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.653,
  latency: 72,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 303,
  confidence: 0.5343,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_011' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.8224,
  latency: 243,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 861,
  confidence: 0.8605,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_012' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.1131,
  latency: 29,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 9796,
  confidence: 0.3734,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_013' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.4967,
  latency: 65,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 9904,
  confidence: 0.8356,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_014' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.2269,
  latency: 227,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 1413,
  confidence: 0.7161,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_015' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.6059,
  latency: 122,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 5300,
  confidence: 0.4272,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_016' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.585,
  latency: 113,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 7085,
  confidence: 0.6213,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_017' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.3832,
  latency: 220,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 8119,
  confidence: 0.6075,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_018' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.0534,
  latency: 246,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9959,
  confidence: 0.735,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_019' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.0373,
  latency: 74,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 3321,
  confidence: 0.8029,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_020' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.5109,
  latency: 23,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 3086,
  confidence: 0.5212,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_021' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.9774,
  latency: 87,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 6639,
  confidence: 0.9592,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_022' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.8669,
  latency: 121,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 1854,
  confidence: 0.3529,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_023' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.4046,
  latency: 242,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 4371,
  confidence: 0.6215,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_024' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.0091,
  latency: 34,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 313,
  confidence: 0.6305,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_025' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.7889,
  latency: 168,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 195,
  confidence: 0.1221,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_026' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_027' })
MERGE (a)-[r_026:MONITORS {
  strength: 0.1605,
  latency: 11,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1922,
  confidence: 0.6819,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_027' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.4565,
  latency: 243,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 6624,
  confidence: 0.3496,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_028' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.2273,
  latency: 63,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 426,
  confidence: 0.5012,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_03_config_managers_2_029' }),
      (b:Explainability { identifier: 'explainability_03_config_managers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.4156,
  latency: 53,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 1921,
  confidence: 0.3682,
  active: true
}]->(b);
