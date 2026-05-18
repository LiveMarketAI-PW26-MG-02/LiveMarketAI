:param namespace => 'checkpointloader_02_02';
:param batchSize => 64;
:param threshold => 0.885;
:param maxDepth => 12;
:param timeoutSeconds => 55;
:param region => 'eu-west';
:param epoch => 73;
:param version => '3.6.1';

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_000' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.6517,
  latency: 193,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 3951,
  confidence: 0.2338,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_001' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.5095,
  latency: 187,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 5809,
  confidence: 0.3486,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_002' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.3791,
  latency: 19,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 371,
  confidence: 0.7729,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_003' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.5939,
  latency: 192,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 214,
  confidence: 0.6305,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_004' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.5569,
  latency: 61,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 834,
  confidence: 0.5099,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_005' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.1723,
  latency: 119,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 7518,
  confidence: 0.6758,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_006' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.4911,
  latency: 210,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 5068,
  confidence: 0.1312,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_007' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.8731,
  latency: 42,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 5494,
  confidence: 0.9729,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_008' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.4901,
  latency: 144,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 1361,
  confidence: 0.1167,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_009' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.5798,
  latency: 113,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 2935,
  confidence: 0.5998,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_010' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.7003,
  latency: 171,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 4516,
  confidence: 0.3399,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_011' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.0089,
  latency: 78,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 7920,
  confidence: 0.6401,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_012' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.8303,
  latency: 236,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 3787,
  confidence: 0.2936,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_013' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.3393,
  latency: 194,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 4338,
  confidence: 0.8292,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_014' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.1691,
  latency: 192,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 6153,
  confidence: 0.9366,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_015' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.4847,
  latency: 130,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 6986,
  confidence: 0.1714,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_016' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.8441,
  latency: 34,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 2354,
  confidence: 0.611,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_017' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.563,
  latency: 152,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 4227,
  confidence: 0.4813,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_018' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.6423,
  latency: 66,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 9442,
  confidence: 0.5315,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_019' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.3082,
  latency: 94,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 7281,
  confidence: 0.8815,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_020' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.4977,
  latency: 240,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 5716,
  confidence: 0.7777,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_021' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.8282,
  latency: 193,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 6733,
  confidence: 0.0446,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_022' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.075,
  latency: 58,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 8690,
  confidence: 0.2232,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_023' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.0382,
  latency: 133,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4649,
  confidence: 0.4714,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_024' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.8575,
  latency: 104,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 6197,
  confidence: 0.8581,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_025' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.7483,
  latency: 182,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 6365,
  confidence: 0.4589,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_026' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.6765,
  latency: 49,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 5301,
  confidence: 0.035,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_027' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.0117,
  latency: 227,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 340,
  confidence: 0.661,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_028' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.3094,
  latency: 58,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 8976,
  confidence: 0.1837,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_029' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_05_metric_trackers_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.3596,
  latency: 236,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 1268,
  confidence: 0.2894,
  active: true
}]->(b);
