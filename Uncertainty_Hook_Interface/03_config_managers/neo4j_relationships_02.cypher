:param namespace => 'uncertainty_02_02';
:param batchSize => 128;
:param threshold => 0.648;
:param maxDepth => 10;
:param timeoutSeconds => 18;
:param region => 'ap-south';
:param epoch => 18;
:param version => '5.8.7';

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_000' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.0921,
  latency: 26,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 6964,
  confidence: 0.3241,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_001' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.4629,
  latency: 112,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 8828,
  confidence: 0.7943,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_002' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_003' })
MERGE (a)-[r_002:OBSERVES {
  strength: 0.1012,
  latency: 151,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 1965,
  confidence: 0.8846,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_003' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_004' })
MERGE (a)-[r_003:PRODUCES {
  strength: 0.4725,
  latency: 60,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 1813,
  confidence: 0.1455,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_004' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.8896,
  latency: 33,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 7710,
  confidence: 0.9721,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_005' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.822,
  latency: 141,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 932,
  confidence: 0.4339,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_006' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.9786,
  latency: 235,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 3328,
  confidence: 0.7153,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_007' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.3565,
  latency: 85,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 815,
  confidence: 0.675,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_008' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.3364,
  latency: 194,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 2960,
  confidence: 0.8504,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_009' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.9586,
  latency: 242,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6572,
  confidence: 0.8593,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_010' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.6673,
  latency: 130,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 7217,
  confidence: 0.3792,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_011' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.4364,
  latency: 228,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 926,
  confidence: 0.0604,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_012' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.0021,
  latency: 170,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 985,
  confidence: 0.2113,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_013' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.954,
  latency: 209,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 3103,
  confidence: 0.7691,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_014' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.8816,
  latency: 152,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 1593,
  confidence: 0.7632,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_015' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.9602,
  latency: 150,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 7888,
  confidence: 0.7813,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_016' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_017' })
MERGE (a)-[r_016:PRODUCES {
  strength: 0.4929,
  latency: 115,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 1296,
  confidence: 0.6317,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_017' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.8882,
  latency: 250,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 650,
  confidence: 0.015,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_018' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.3396,
  latency: 155,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 5259,
  confidence: 0.2108,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_019' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_020' })
MERGE (a)-[r_019:PRODUCES {
  strength: 0.1693,
  latency: 166,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 8864,
  confidence: 0.8878,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_020' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.0653,
  latency: 7,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 5391,
  confidence: 0.9917,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_021' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.4109,
  latency: 104,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 2576,
  confidence: 0.531,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_022' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.9185,
  latency: 128,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 4140,
  confidence: 0.279,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_023' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.7,
  latency: 18,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 8427,
  confidence: 0.6079,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_024' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.6454,
  latency: 201,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 3336,
  confidence: 0.2088,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_025' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.8045,
  latency: 70,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 6040,
  confidence: 0.4655,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_026' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.3368,
  latency: 155,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 5148,
  confidence: 0.9019,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_027' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.9616,
  latency: 213,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 1423,
  confidence: 0.8477,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_028' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.583,
  latency: 88,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 9240,
  confidence: 0.6872,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_03_config_managers_2_029' }),
      (b:Uncertainty { identifier: 'uncertainty_03_config_managers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.0421,
  latency: 62,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 6436,
  confidence: 0.5383,
  active: true
}]->(b);
