:param namespace => 'graphnetwork_02_02';
:param batchSize => 32;
:param threshold => 0.243;
:param maxDepth => 10;
:param timeoutSeconds => 58;
:param region => 'ap-south';
:param epoch => 3;
:param version => '3.1.7';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.5529,
  latency: 138,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 3111,
  confidence: 0.2218,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.6443,
  latency: 70,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 2864,
  confidence: 0.7203,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.9823,
  latency: 33,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 8563,
  confidence: 0.3924,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.6577,
  latency: 223,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 336,
  confidence: 0.2254,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.2195,
  latency: 157,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 7775,
  confidence: 0.1214,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.9311,
  latency: 105,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 1329,
  confidence: 0.3645,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.1347,
  latency: 215,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 1460,
  confidence: 0.9276,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.2366,
  latency: 139,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 6277,
  confidence: 0.5362,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.9249,
  latency: 176,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 4803,
  confidence: 0.4718,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.7729,
  latency: 170,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 537,
  confidence: 0.9608,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.1105,
  latency: 94,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 5991,
  confidence: 0.5523,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.2846,
  latency: 53,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 3025,
  confidence: 0.4121,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.1521,
  latency: 108,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 9803,
  confidence: 0.0736,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.2722,
  latency: 179,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 1592,
  confidence: 0.6738,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.2711,
  latency: 113,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 9802,
  confidence: 0.6036,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.1618,
  latency: 87,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 8951,
  confidence: 0.1464,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_017' })
MERGE (a)-[r_016:TRIGGERS {
  strength: 0.7789,
  latency: 172,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 5341,
  confidence: 0.5548,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.0437,
  latency: 244,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 1461,
  confidence: 0.1397,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.9866,
  latency: 84,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 2714,
  confidence: 0.0462,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.5078,
  latency: 149,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 197,
  confidence: 0.4416,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.8865,
  latency: 26,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8768,
  confidence: 0.8214,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.1453,
  latency: 211,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 3875,
  confidence: 0.7603,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.351,
  latency: 79,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 6109,
  confidence: 0.4124,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.3604,
  latency: 86,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 5981,
  confidence: 0.4257,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.5693,
  latency: 180,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 6644,
  confidence: 0.2409,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.5791,
  latency: 161,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 5145,
  confidence: 0.9726,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.6423,
  latency: 179,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6427,
  confidence: 0.5471,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.1431,
  latency: 129,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 8237,
  confidence: 0.7763,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.0573,
  latency: 250,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 6792,
  confidence: 0.854,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_03_config_managers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.5876,
  latency: 113,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 8262,
  confidence: 0.0708,
  active: true
}]->(b);
