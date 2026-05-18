:param namespace => 'alignment_02_02';
:param batchSize => 256;
:param threshold => 0.521;
:param maxDepth => 5;
:param timeoutSeconds => 88;
:param region => 'eu-west';
:param epoch => 21;
:param version => '3.9.1';

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_000' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.9719,
  latency: 9,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 6985,
  confidence: 0.0431,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_001' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.6558,
  latency: 222,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 9186,
  confidence: 0.1742,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_002' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.0139,
  latency: 72,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 6462,
  confidence: 0.9703,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_003' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.308,
  latency: 97,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 4632,
  confidence: 0.5964,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_004' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.1742,
  latency: 79,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 5560,
  confidence: 0.2891,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_005' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.2748,
  latency: 33,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 6670,
  confidence: 0.8071,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_006' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.4333,
  latency: 199,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 9989,
  confidence: 0.913,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_007' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.2575,
  latency: 67,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 7805,
  confidence: 0.0385,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_008' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.2549,
  latency: 17,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 7912,
  confidence: 0.8038,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_009' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.9521,
  latency: 154,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 2319,
  confidence: 0.0362,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_010' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.6989,
  latency: 83,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 1324,
  confidence: 0.7661,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_011' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.6219,
  latency: 142,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 9345,
  confidence: 0.5602,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_012' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.5662,
  latency: 250,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 7037,
  confidence: 0.7967,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_013' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.1432,
  latency: 166,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 9245,
  confidence: 0.8803,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_014' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.2812,
  latency: 121,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 8646,
  confidence: 0.4195,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_015' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.9254,
  latency: 116,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 9823,
  confidence: 0.0456,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_016' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.827,
  latency: 161,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 3577,
  confidence: 0.0677,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_017' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.2016,
  latency: 9,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 2135,
  confidence: 0.1841,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_018' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.1346,
  latency: 38,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 7276,
  confidence: 0.0377,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_019' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.1067,
  latency: 77,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 4293,
  confidence: 0.5891,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_020' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.3747,
  latency: 99,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 3765,
  confidence: 0.9493,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_021' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.9114,
  latency: 246,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 4982,
  confidence: 0.1865,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_022' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.5727,
  latency: 55,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 6375,
  confidence: 0.571,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_023' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.4927,
  latency: 241,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 335,
  confidence: 0.977,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_024' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.3345,
  latency: 44,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 4138,
  confidence: 0.2069,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_025' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.0779,
  latency: 89,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6748,
  confidence: 0.9495,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_026' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.5868,
  latency: 82,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 8872,
  confidence: 0.4532,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_027' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.4934,
  latency: 5,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 9345,
  confidence: 0.8989,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_028' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.6981,
  latency: 17,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 4718,
  confidence: 0.3727,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_01_core_engine_2_029' }),
      (b:Alignment { identifier: 'alignment_01_core_engine_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.7647,
  latency: 170,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 177,
  confidence: 0.8694,
  active: true
}]->(b);
