:param namespace => 'transformer_02_02';
:param batchSize => 256;
:param threshold => 0.212;
:param maxDepth => 12;
:param timeoutSeconds => 12;
:param region => 'us-west';
:param epoch => 15;
:param version => '4.7.0';

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_000' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.5779,
  latency: 187,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 6993,
  confidence: 0.0123,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_001' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_002' })
MERGE (a)-[r_001:VALIDATES {
  strength: 0.8576,
  latency: 186,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 9391,
  confidence: 0.7452,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_002' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.6015,
  latency: 250,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 3052,
  confidence: 0.7257,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_003' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.7551,
  latency: 114,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 4789,
  confidence: 0.4831,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_004' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.9699,
  latency: 63,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 6394,
  confidence: 0.6573,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_005' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_006' })
MERGE (a)-[r_005:OBSERVES {
  strength: 0.3,
  latency: 170,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 1817,
  confidence: 0.6983,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_006' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.7616,
  latency: 85,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 5884,
  confidence: 0.0421,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_007' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.0607,
  latency: 183,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 7950,
  confidence: 0.7656,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_008' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.698,
  latency: 196,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 2153,
  confidence: 0.7486,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_009' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.2342,
  latency: 111,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 3897,
  confidence: 0.6771,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_010' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.9356,
  latency: 238,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 9140,
  confidence: 0.6693,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_011' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_012' })
MERGE (a)-[r_011:DEPENDS_ON {
  strength: 0.2346,
  latency: 145,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 1087,
  confidence: 0.5152,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_012' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.3808,
  latency: 52,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 180,
  confidence: 0.3492,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_013' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.893,
  latency: 107,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 4018,
  confidence: 0.2836,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_014' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.1345,
  latency: 144,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 2787,
  confidence: 0.9862,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_015' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.3525,
  latency: 170,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 8210,
  confidence: 0.6049,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_016' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.856,
  latency: 137,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 8801,
  confidence: 0.5967,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_017' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.086,
  latency: 66,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 5296,
  confidence: 0.5597,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_018' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.724,
  latency: 184,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 5138,
  confidence: 0.4645,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_019' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.8999,
  latency: 208,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 3550,
  confidence: 0.492,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_020' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.6409,
  latency: 9,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 9100,
  confidence: 0.1841,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_021' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.0279,
  latency: 55,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 8192,
  confidence: 0.0138,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_022' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.655,
  latency: 34,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 2357,
  confidence: 0.5431,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_023' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.0566,
  latency: 234,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 2717,
  confidence: 0.1917,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_024' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.4809,
  latency: 40,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 1276,
  confidence: 0.3362,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_025' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.256,
  latency: 186,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 4743,
  confidence: 0.7829,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_026' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.8282,
  latency: 36,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 3573,
  confidence: 0.5762,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_027' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.2338,
  latency: 128,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 5870,
  confidence: 0.5663,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_028' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_029' })
MERGE (a)-[r_028:ROUTES_TO {
  strength: 0.6743,
  latency: 86,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7300,
  confidence: 0.442,
  active: true
}]->(b);

MATCH (a:Transformer { identifier: 'transformer_05_metric_trackers_2_029' }),
      (b:Transformer { identifier: 'transformer_05_metric_trackers_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.2217,
  latency: 170,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 6658,
  confidence: 0.8027,
  active: true
}]->(b);
