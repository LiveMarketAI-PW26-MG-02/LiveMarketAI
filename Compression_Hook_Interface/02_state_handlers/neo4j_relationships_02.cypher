:param namespace => 'compression_02_02';
:param batchSize => 32;
:param threshold => 0.509;
:param maxDepth => 10;
:param timeoutSeconds => 99;
:param region => 'eu-west';
:param epoch => 58;
:param version => '4.6.7';

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_000' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_001' })
MERGE (a)-[r_000:PRODUCES {
  strength: 0.6684,
  latency: 90,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 7319,
  confidence: 0.3302,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_001' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.7391,
  latency: 74,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 5292,
  confidence: 0.8165,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_002' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.1753,
  latency: 169,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 5776,
  confidence: 0.5923,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_003' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.4318,
  latency: 115,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 3976,
  confidence: 0.6545,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_004' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.1078,
  latency: 98,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7872,
  confidence: 0.3636,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_005' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.3162,
  latency: 181,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 1542,
  confidence: 0.4913,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_006' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.4523,
  latency: 232,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 783,
  confidence: 0.014,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_007' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.6065,
  latency: 31,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 7942,
  confidence: 0.5828,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_008' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.7946,
  latency: 236,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 9363,
  confidence: 0.9781,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_009' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.9661,
  latency: 239,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 8818,
  confidence: 0.7978,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_010' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.844,
  latency: 208,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 1715,
  confidence: 0.9482,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_011' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.2131,
  latency: 34,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 5065,
  confidence: 0.3614,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_012' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.6796,
  latency: 38,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 7950,
  confidence: 0.1344,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_013' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.4938,
  latency: 136,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 7284,
  confidence: 0.056,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_014' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.4297,
  latency: 77,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 6013,
  confidence: 0.0427,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_015' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.4233,
  latency: 49,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 3112,
  confidence: 0.3985,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_016' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.0439,
  latency: 48,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 2581,
  confidence: 0.6291,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_017' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_018' })
MERGE (a)-[r_017:PRODUCES {
  strength: 0.7526,
  latency: 54,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 1112,
  confidence: 0.4001,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_018' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.352,
  latency: 11,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6057,
  confidence: 0.5012,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_019' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.0547,
  latency: 69,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 8898,
  confidence: 0.6597,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_020' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.868,
  latency: 224,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 432,
  confidence: 0.2623,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_021' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.5231,
  latency: 25,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 5389,
  confidence: 0.2423,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_022' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.6656,
  latency: 173,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 8271,
  confidence: 0.0378,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_023' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_024' })
MERGE (a)-[r_023:ROUTES_TO {
  strength: 0.5216,
  latency: 45,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 180,
  confidence: 0.7004,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_024' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.943,
  latency: 57,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 8501,
  confidence: 0.8971,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_025' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.1118,
  latency: 177,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 1660,
  confidence: 0.2553,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_026' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_027' })
MERGE (a)-[r_026:MONITORS {
  strength: 0.1815,
  latency: 107,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 5693,
  confidence: 0.546,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_027' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.6477,
  latency: 209,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 912,
  confidence: 0.0227,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_028' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.9328,
  latency: 245,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 9224,
  confidence: 0.6626,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_02_state_handlers_2_029' }),
      (b:Compression { identifier: 'compression_02_state_handlers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.9428,
  latency: 39,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 2952,
  confidence: 0.0323,
  active: true
}]->(b);
