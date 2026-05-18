:param namespace => 'tabularmodel_02_02';
:param batchSize => 32;
:param threshold => 0.503;
:param maxDepth => 6;
:param timeoutSeconds => 12;
:param region => 'eu-west';
:param epoch => 92;
:param version => '4.3.1';

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_000' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.2937,
  latency: 70,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 7076,
  confidence: 0.8929,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_001' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.0023,
  latency: 1,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 3716,
  confidence: 0.7462,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_002' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.7337,
  latency: 22,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 938,
  confidence: 0.6927,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_003' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.7388,
  latency: 151,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 5523,
  confidence: 0.0308,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_004' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.4647,
  latency: 87,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 2678,
  confidence: 0.4459,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_005' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.3749,
  latency: 33,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 9658,
  confidence: 0.3651,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_006' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.363,
  latency: 6,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 6971,
  confidence: 0.8488,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_007' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.4395,
  latency: 163,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 1377,
  confidence: 0.6123,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_008' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.5692,
  latency: 149,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 5567,
  confidence: 0.0919,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_009' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.0312,
  latency: 70,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 5101,
  confidence: 0.8288,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_010' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.9187,
  latency: 180,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 4686,
  confidence: 0.7137,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_011' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.6988,
  latency: 120,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 6863,
  confidence: 0.6711,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_012' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.8745,
  latency: 7,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 1586,
  confidence: 0.5808,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_013' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.9667,
  latency: 180,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 4389,
  confidence: 0.6875,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_014' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_015' })
MERGE (a)-[r_014:TRIGGERS {
  strength: 0.8951,
  latency: 86,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 8136,
  confidence: 0.1905,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_015' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.8246,
  latency: 70,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 1038,
  confidence: 0.8916,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_016' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.0251,
  latency: 134,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 5951,
  confidence: 0.5176,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_017' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.7226,
  latency: 3,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 5816,
  confidence: 0.9541,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_018' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.5646,
  latency: 28,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 1345,
  confidence: 0.2459,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_019' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.7167,
  latency: 22,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 4228,
  confidence: 0.1063,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_020' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.2624,
  latency: 186,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 7202,
  confidence: 0.1169,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_021' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.9836,
  latency: 154,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 3661,
  confidence: 0.6978,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_022' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.0357,
  latency: 95,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 3289,
  confidence: 0.0597,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_023' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_024' })
MERGE (a)-[r_023:ROUTES_TO {
  strength: 0.5051,
  latency: 140,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 7484,
  confidence: 0.8911,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_024' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.6014,
  latency: 239,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 1821,
  confidence: 0.5639,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_025' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.9262,
  latency: 7,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 6801,
  confidence: 0.679,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_026' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.1577,
  latency: 56,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 1879,
  confidence: 0.2772,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_027' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.3872,
  latency: 183,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 1373,
  confidence: 0.1286,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_028' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.5584,
  latency: 124,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 4163,
  confidence: 0.8464,
  active: true
}]->(b);

MATCH (a:TabularModel { identifier: 'tabularmodel_01_core_engine_2_029' }),
      (b:TabularModel { identifier: 'tabularmodel_01_core_engine_2_000' })
MERGE (a)-[r_029:VALIDATES {
  strength: 0.9337,
  latency: 173,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 6144,
  confidence: 0.8246,
  active: true
}]->(b);
