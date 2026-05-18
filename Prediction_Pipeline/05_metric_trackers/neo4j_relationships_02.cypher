:param namespace => 'predictionpipeline_02_02';
:param batchSize => 32;
:param threshold => 0.727;
:param maxDepth => 11;
:param timeoutSeconds => 89;
:param region => 'eu-west';
:param epoch => 95;
:param version => '1.2.3';

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_000' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.2563,
  latency: 57,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 1873,
  confidence: 0.078,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_001' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.4514,
  latency: 51,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1178,
  confidence: 0.8417,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_002' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.1486,
  latency: 221,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 8592,
  confidence: 0.7932,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_003' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.8989,
  latency: 91,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 396,
  confidence: 0.2225,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_004' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.2742,
  latency: 225,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 850,
  confidence: 0.0692,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_005' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.8981,
  latency: 190,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 9879,
  confidence: 0.3219,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_006' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.7563,
  latency: 110,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 5654,
  confidence: 0.131,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_007' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.0397,
  latency: 29,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 9551,
  confidence: 0.0382,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_008' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.9185,
  latency: 225,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 4665,
  confidence: 0.6156,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_009' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.5956,
  latency: 223,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 8817,
  confidence: 0.0056,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_010' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.6978,
  latency: 240,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4371,
  confidence: 0.1888,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_011' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_012' })
MERGE (a)-[r_011:DEPENDS_ON {
  strength: 0.8727,
  latency: 7,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 7296,
  confidence: 0.5589,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_012' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.3392,
  latency: 200,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 2695,
  confidence: 0.676,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_013' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.5513,
  latency: 176,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 246,
  confidence: 0.5694,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_014' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.0732,
  latency: 221,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 9542,
  confidence: 0.7192,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_015' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.7607,
  latency: 65,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 7026,
  confidence: 0.2821,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_016' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.6556,
  latency: 61,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 6075,
  confidence: 0.43,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_017' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.6476,
  latency: 200,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 9109,
  confidence: 0.1477,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_018' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.2826,
  latency: 20,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 5616,
  confidence: 0.5151,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_019' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.7591,
  latency: 243,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 6481,
  confidence: 0.5763,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_020' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.9021,
  latency: 156,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 7123,
  confidence: 0.4489,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_021' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.6958,
  latency: 56,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 1627,
  confidence: 0.2764,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_022' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.9279,
  latency: 29,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 9753,
  confidence: 0.0494,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_023' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.8752,
  latency: 96,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 9121,
  confidence: 0.5435,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_024' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.8441,
  latency: 20,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 2588,
  confidence: 0.5182,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_025' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.6501,
  latency: 156,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 4668,
  confidence: 0.6828,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_026' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.4538,
  latency: 165,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 1231,
  confidence: 0.2854,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_027' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.8107,
  latency: 51,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 456,
  confidence: 0.0008,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_028' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.5942,
  latency: 223,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 5335,
  confidence: 0.1821,
  active: true
}]->(b);

MATCH (a:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_029' }),
      (b:PredictionPipeline { identifier: 'predictionpipeline_05_metric_trackers_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.6284,
  latency: 204,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 5000,
  confidence: 0.2305,
  active: true
}]->(b);
