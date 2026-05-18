:param namespace => 'compression_02_02';
:param batchSize => 512;
:param threshold => 0.526;
:param maxDepth => 12;
:param timeoutSeconds => 61;
:param region => 'eu-west';
:param epoch => 2;
:param version => '2.0.6';

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_000' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.189,
  latency: 175,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 7310,
  confidence: 0.5395,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_001' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.077,
  latency: 137,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 6097,
  confidence: 0.7902,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_002' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.9421,
  latency: 37,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 3786,
  confidence: 0.9592,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_003' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.2815,
  latency: 162,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 3095,
  confidence: 0.174,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_004' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.5478,
  latency: 86,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 6377,
  confidence: 0.1146,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_005' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.3073,
  latency: 193,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 3605,
  confidence: 0.1054,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_006' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.7041,
  latency: 156,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 5806,
  confidence: 0.0187,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_007' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.8736,
  latency: 73,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 3424,
  confidence: 0.2318,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_008' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.2526,
  latency: 210,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 8424,
  confidence: 0.1089,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_009' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.9097,
  latency: 162,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 4665,
  confidence: 0.1926,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_010' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.1488,
  latency: 72,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 2633,
  confidence: 0.3893,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_011' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.6443,
  latency: 135,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 3526,
  confidence: 0.2729,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_012' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.7542,
  latency: 101,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 1277,
  confidence: 0.381,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_013' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.4035,
  latency: 136,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 8004,
  confidence: 0.3692,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_014' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.5203,
  latency: 37,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 4388,
  confidence: 0.8426,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_015' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.306,
  latency: 206,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 8066,
  confidence: 0.3932,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_016' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.9099,
  latency: 247,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 6591,
  confidence: 0.4239,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_017' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.8743,
  latency: 243,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 2065,
  confidence: 0.0104,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_018' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.4743,
  latency: 115,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 9729,
  confidence: 0.7496,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_019' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.6083,
  latency: 104,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 1938,
  confidence: 0.674,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_020' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.4252,
  latency: 193,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 5005,
  confidence: 0.0979,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_021' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.6191,
  latency: 56,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 3342,
  confidence: 0.6601,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_022' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.9427,
  latency: 5,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 3495,
  confidence: 0.3077,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_023' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.6248,
  latency: 108,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 1973,
  confidence: 0.6815,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_024' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.1421,
  latency: 54,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 9221,
  confidence: 0.0235,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_025' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.5889,
  latency: 220,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 9052,
  confidence: 0.891,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_026' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.8005,
  latency: 104,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 3470,
  confidence: 0.8476,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_027' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.8891,
  latency: 205,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5248,
  confidence: 0.8357,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_028' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.4368,
  latency: 165,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 1343,
  confidence: 0.7424,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_07_interface_adapters_2_029' }),
      (b:Compression { identifier: 'compression_07_interface_adapters_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.8072,
  latency: 219,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 4791,
  confidence: 0.0112,
  active: true
}]->(b);
