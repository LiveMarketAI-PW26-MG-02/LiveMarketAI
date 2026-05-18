:param namespace => 'predictionpipeline_02_02';
:param batchSize => 128;
:param threshold => 0.322;
:param maxDepth => 4;
:param timeoutSeconds => 25;
:param region => 'ap-south';
:param epoch => 2;
:param version => '2.1.3';

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_000' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.0476,
  latency: 51,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 8063,
  confidence: 0.444,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_001' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.5886,
  latency: 159,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 3289,
  confidence: 0.8263,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_002' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.6426,
  latency: 226,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 2517,
  confidence: 0.6455,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_003' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.7572,
  latency: 25,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 209,
  confidence: 0.8953,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_004' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.7994,
  latency: 212,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 5324,
  confidence: 0.9951,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_005' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.0845,
  latency: 82,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 5774,
  confidence: 0.4435,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_006' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.918,
  latency: 211,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 5128,
  confidence: 0.2353,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_007' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.8264,
  latency: 169,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 5850,
  confidence: 0.1493,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_008' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.0494,
  latency: 36,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 2626,
  confidence: 0.3035,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_009' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.4148,
  latency: 140,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 2014,
  confidence: 0.0074,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_010' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.3081,
  latency: 138,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 910,
  confidence: 0.8916,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_011' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.8462,
  latency: 49,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 5961,
  confidence: 0.6298,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_012' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.3006,
  latency: 250,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 3987,
  confidence: 0.5567,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_013' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.7349,
  latency: 96,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 1765,
  confidence: 0.2323,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_014' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.5971,
  latency: 104,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 8913,
  confidence: 0.1787,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_015' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.6209,
  latency: 178,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 8450,
  confidence: 0.8619,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_016' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.8787,
  latency: 160,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 2168,
  confidence: 0.3427,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_017' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.3409,
  latency: 156,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 9849,
  confidence: 0.3557,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_018' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.9602,
  latency: 54,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 2569,
  confidence: 0.9707,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_019' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.456,
  latency: 178,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 6754,
  confidence: 0.256,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_020' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.8292,
  latency: 103,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 4326,
  confidence: 0.7065,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_021' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.353,
  latency: 65,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 8300,
  confidence: 0.3919,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_022' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.4089,
  latency: 240,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 5890,
  confidence: 0.298,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_023' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.7048,
  latency: 189,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 3224,
  confidence: 0.1217,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_024' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.907,
  latency: 230,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 1694,
  confidence: 0.8769,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_025' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.6425,
  latency: 28,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 2802,
  confidence: 0.2621,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_026' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.9593,
  latency: 211,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 998,
  confidence: 0.2304,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_027' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.4872,
  latency: 46,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 9307,
  confidence: 0.5193,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_028' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.1243,
  latency: 16,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 4917,
  confidence: 0.5496,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_029' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_03_config_managers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.506,
  latency: 192,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 5011,
  confidence: 0.4992,
  active: true
}]->(b);
