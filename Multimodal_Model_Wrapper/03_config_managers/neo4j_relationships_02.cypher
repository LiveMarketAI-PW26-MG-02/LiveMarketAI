:param namespace => 'multimodal_02_02';
:param batchSize => 64;
:param threshold => 0.74;
:param maxDepth => 10;
:param timeoutSeconds => 17;
:param region => 'us-east';
:param epoch => 60;
:param version => '4.8.5';

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_000' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.672,
  latency: 76,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 7576,
  confidence: 0.1781,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_001' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.1013,
  latency: 32,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 2804,
  confidence: 0.6739,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_002' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.2265,
  latency: 163,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 5557,
  confidence: 0.8618,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_003' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.393,
  latency: 188,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 919,
  confidence: 0.458,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_004' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.5722,
  latency: 100,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 4381,
  confidence: 0.1662,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_005' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.6763,
  latency: 184,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 2034,
  confidence: 0.2365,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_006' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.7708,
  latency: 186,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 2813,
  confidence: 0.665,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_007' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.7705,
  latency: 226,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 1183,
  confidence: 0.4908,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_008' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.3504,
  latency: 197,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 3930,
  confidence: 0.1459,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_009' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.2373,
  latency: 6,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 2644,
  confidence: 0.6309,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_010' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.2645,
  latency: 61,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1819,
  confidence: 0.9981,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_011' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.3114,
  latency: 173,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 6312,
  confidence: 0.2552,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_012' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.6195,
  latency: 59,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 8928,
  confidence: 0.506,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_013' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.9719,
  latency: 220,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 5327,
  confidence: 0.1131,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_014' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.2369,
  latency: 13,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 4757,
  confidence: 0.5673,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_015' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.3106,
  latency: 188,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 1571,
  confidence: 0.9932,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_016' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.971,
  latency: 231,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 3680,
  confidence: 0.1588,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_017' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.79,
  latency: 95,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 7281,
  confidence: 0.2903,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_018' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.3803,
  latency: 195,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 5056,
  confidence: 0.2259,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_019' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.2564,
  latency: 101,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6160,
  confidence: 0.9827,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_020' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.7775,
  latency: 160,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 7630,
  confidence: 0.7681,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_021' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.0675,
  latency: 191,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 5256,
  confidence: 0.4125,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_022' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.839,
  latency: 84,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 6896,
  confidence: 0.7458,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_023' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.4396,
  latency: 59,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 2448,
  confidence: 0.0682,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_024' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.2754,
  latency: 114,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 9626,
  confidence: 0.526,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_025' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.1347,
  latency: 147,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 8212,
  confidence: 0.9439,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_026' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.5232,
  latency: 191,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 9989,
  confidence: 0.884,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_027' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.8066,
  latency: 132,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 6411,
  confidence: 0.6596,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_028' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.1268,
  latency: 139,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 6686,
  confidence: 0.757,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_03_config_managers_2_029' }),
      (b:Multimodal { identifier: 'multimodal_03_config_managers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.5359,
  latency: 29,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 9355,
  confidence: 0.4087,
  active: true
}]->(b);
