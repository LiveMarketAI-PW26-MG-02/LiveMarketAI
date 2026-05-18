:param namespace => 'checkpointloader_02_02';
:param batchSize => 512;
:param threshold => 0.707;
:param maxDepth => 7;
:param timeoutSeconds => 80;
:param region => 'eu-west';
:param epoch => 20;
:param version => '4.1.2';

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_000' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.7694,
  latency: 51,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 5638,
  confidence: 0.8443,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_001' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.9664,
  latency: 159,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 2016,
  confidence: 0.7218,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_002' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.1411,
  latency: 233,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6632,
  confidence: 0.7889,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_003' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.5214,
  latency: 169,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 1915,
  confidence: 0.6323,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_004' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.8235,
  latency: 160,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 6175,
  confidence: 0.4431,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_005' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.2504,
  latency: 43,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 2020,
  confidence: 0.2794,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_006' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.5093,
  latency: 211,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 4720,
  confidence: 0.6608,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_007' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.5578,
  latency: 169,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 4535,
  confidence: 0.4581,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_008' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.6174,
  latency: 209,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 2111,
  confidence: 0.0293,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_009' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.5205,
  latency: 67,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 3339,
  confidence: 0.0655,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_010' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.2608,
  latency: 45,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 9265,
  confidence: 0.2221,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_011' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.5908,
  latency: 242,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 8629,
  confidence: 0.3784,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_012' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.3266,
  latency: 103,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 7085,
  confidence: 0.7433,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_013' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.167,
  latency: 145,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 4149,
  confidence: 0.5222,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_014' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.4645,
  latency: 234,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 4379,
  confidence: 0.0703,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_015' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.0499,
  latency: 245,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4636,
  confidence: 0.7406,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_016' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.1343,
  latency: 48,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 9603,
  confidence: 0.4017,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_017' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.8512,
  latency: 147,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 4853,
  confidence: 0.9682,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_018' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_019' })
MERGE (a)-[r_018:CALIBRATES {
  strength: 0.5255,
  latency: 234,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 6856,
  confidence: 0.4419,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_019' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.8185,
  latency: 9,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 6504,
  confidence: 0.1962,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_020' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.9172,
  latency: 110,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 2880,
  confidence: 0.8102,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_021' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.4307,
  latency: 210,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 2992,
  confidence: 0.0057,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_022' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.0006,
  latency: 53,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 1123,
  confidence: 0.9421,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_023' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.1949,
  latency: 9,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 1960,
  confidence: 0.3388,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_024' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.7672,
  latency: 47,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 1180,
  confidence: 0.6066,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_025' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.9177,
  latency: 6,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 9842,
  confidence: 0.1819,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_026' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.934,
  latency: 2,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 187,
  confidence: 0.3689,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_027' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.44,
  latency: 184,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 1689,
  confidence: 0.7229,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_028' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.1823,
  latency: 204,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 1344,
  confidence: 0.7677,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_029' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_01_core_engine_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.7887,
  latency: 59,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 4647,
  confidence: 0.6246,
  active: true
}]->(b);
