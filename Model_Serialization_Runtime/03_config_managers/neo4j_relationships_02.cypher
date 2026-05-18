:param namespace => 'serializer_02_02';
:param batchSize => 256;
:param threshold => 0.696;
:param maxDepth => 11;
:param timeoutSeconds => 10;
:param region => 'ap-south';
:param epoch => 78;
:param version => '5.1.0';

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_000' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.1248,
  latency: 103,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 1608,
  confidence: 0.6199,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_001' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.9261,
  latency: 62,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 6492,
  confidence: 0.7741,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_002' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.7733,
  latency: 64,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 3323,
  confidence: 0.8115,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_003' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.7401,
  latency: 174,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 9929,
  confidence: 0.8502,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_004' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.5053,
  latency: 160,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 6501,
  confidence: 0.1001,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_005' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.629,
  latency: 182,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 648,
  confidence: 0.9798,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_006' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.7351,
  latency: 84,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 3639,
  confidence: 0.1147,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_007' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.4763,
  latency: 132,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 6384,
  confidence: 0.261,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_008' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.1377,
  latency: 104,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 2769,
  confidence: 0.2503,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_009' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.8116,
  latency: 127,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 6120,
  confidence: 0.8253,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_010' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.6934,
  latency: 50,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 7021,
  confidence: 0.2092,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_011' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.0036,
  latency: 142,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 8813,
  confidence: 0.775,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_012' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.7489,
  latency: 8,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 4971,
  confidence: 0.2713,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_013' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_014' })
MERGE (a)-[r_013:MONITORS {
  strength: 0.326,
  latency: 26,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 1445,
  confidence: 0.3557,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_014' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.5899,
  latency: 104,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 3070,
  confidence: 0.9352,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_015' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.3029,
  latency: 59,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 9752,
  confidence: 0.5247,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_016' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.4691,
  latency: 16,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 5080,
  confidence: 0.7238,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_017' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_018' })
MERGE (a)-[r_017:PRODUCES {
  strength: 0.4013,
  latency: 194,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 7000,
  confidence: 0.4818,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_018' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.9922,
  latency: 3,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 769,
  confidence: 0.3016,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_019' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_020' })
MERGE (a)-[r_019:CALIBRATES {
  strength: 0.6515,
  latency: 133,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 4961,
  confidence: 0.3804,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_020' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.7341,
  latency: 34,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 9084,
  confidence: 0.5718,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_021' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.8887,
  latency: 20,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 821,
  confidence: 0.0051,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_022' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.9369,
  latency: 100,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 3136,
  confidence: 0.0328,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_023' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.9957,
  latency: 21,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 9500,
  confidence: 0.957,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_024' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.248,
  latency: 206,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 1531,
  confidence: 0.6406,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_025' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.9086,
  latency: 250,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6650,
  confidence: 0.0029,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_026' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.5951,
  latency: 39,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 9676,
  confidence: 0.5481,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_027' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.7186,
  latency: 163,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 3059,
  confidence: 0.0961,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_028' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.3246,
  latency: 2,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 1387,
  confidence: 0.2458,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_03_config_managers_2_029' }),
      (b:Serializer { identifier: 'serializer_03_config_managers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.2767,
  latency: 167,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 9926,
  confidence: 0.2398,
  active: true
}]->(b);
