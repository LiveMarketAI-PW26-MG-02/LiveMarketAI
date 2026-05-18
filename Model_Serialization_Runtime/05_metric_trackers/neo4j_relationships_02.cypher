:param namespace => 'serializer_02_02';
:param batchSize => 128;
:param threshold => 0.16;
:param maxDepth => 11;
:param timeoutSeconds => 42;
:param region => 'eu-west';
:param epoch => 35;
:param version => '1.6.1';

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_000' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.1732,
  latency: 31,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 2742,
  confidence: 0.385,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_001' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.6043,
  latency: 98,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 9564,
  confidence: 0.7659,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_002' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.7094,
  latency: 131,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 8763,
  confidence: 0.4471,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_003' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.9505,
  latency: 148,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6646,
  confidence: 0.949,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_004' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.4593,
  latency: 12,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 9213,
  confidence: 0.6693,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_005' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.6856,
  latency: 200,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 1085,
  confidence: 0.0447,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_006' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.5042,
  latency: 85,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 4328,
  confidence: 0.5158,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_007' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.4727,
  latency: 90,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 1348,
  confidence: 0.4561,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_008' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.4171,
  latency: 148,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5792,
  confidence: 0.8184,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_009' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.2195,
  latency: 54,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 1499,
  confidence: 0.261,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_010' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.467,
  latency: 244,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 5491,
  confidence: 0.0109,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_011' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.6856,
  latency: 16,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1810,
  confidence: 0.2497,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_012' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.3424,
  latency: 203,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 2932,
  confidence: 0.0801,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_013' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.0389,
  latency: 6,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 2798,
  confidence: 0.7348,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_014' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.8489,
  latency: 86,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 4681,
  confidence: 0.4373,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_015' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.2348,
  latency: 2,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 5816,
  confidence: 0.1087,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_016' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.5419,
  latency: 236,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 8280,
  confidence: 0.7859,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_017' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.558,
  latency: 30,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 9066,
  confidence: 0.6318,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_018' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.7984,
  latency: 223,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4433,
  confidence: 0.5614,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_019' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.0887,
  latency: 208,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 7707,
  confidence: 0.7004,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_020' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.399,
  latency: 98,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 4207,
  confidence: 0.0496,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_021' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.0188,
  latency: 212,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 2602,
  confidence: 0.2114,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_022' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.8952,
  latency: 178,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 2040,
  confidence: 0.4239,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_023' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_024' })
MERGE (a)-[r_023:ROUTES_TO {
  strength: 0.4416,
  latency: 82,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 7501,
  confidence: 0.0847,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_024' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.6108,
  latency: 30,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5302,
  confidence: 0.3099,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_025' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.3423,
  latency: 54,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 4696,
  confidence: 0.2002,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_026' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.2892,
  latency: 164,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 1277,
  confidence: 0.6838,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_027' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.9484,
  latency: 179,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 8363,
  confidence: 0.2903,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_028' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.8873,
  latency: 155,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 6727,
  confidence: 0.4255,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_05_metric_trackers_2_029' }),
      (b:Serializer { identifier: 'serializer_05_metric_trackers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.9596,
  latency: 189,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 5999,
  confidence: 0.1041,
  active: true
}]->(b);
