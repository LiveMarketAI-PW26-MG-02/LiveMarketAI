:param namespace => 'compression_02_02';
:param batchSize => 128;
:param threshold => 0.794;
:param maxDepth => 4;
:param timeoutSeconds => 47;
:param region => 'us-east';
:param epoch => 15;
:param version => '3.8.1';

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_000' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.5522,
  latency: 143,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 2690,
  confidence: 0.9878,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_001' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.8998,
  latency: 39,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 5366,
  confidence: 0.1435,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_002' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.4262,
  latency: 159,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 6402,
  confidence: 0.1255,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_003' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.1721,
  latency: 186,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 2202,
  confidence: 0.0617,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_004' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.7692,
  latency: 49,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 1700,
  confidence: 0.2381,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_005' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.7648,
  latency: 81,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 7812,
  confidence: 0.6424,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_006' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.3724,
  latency: 92,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 3633,
  confidence: 0.2408,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_007' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.1672,
  latency: 39,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 9647,
  confidence: 0.8446,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_008' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.7217,
  latency: 58,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 671,
  confidence: 0.398,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_009' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.4722,
  latency: 94,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 7904,
  confidence: 0.1879,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_010' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.5549,
  latency: 15,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 424,
  confidence: 0.3804,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_011' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.4085,
  latency: 182,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 7909,
  confidence: 0.0845,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_012' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.3679,
  latency: 248,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 7353,
  confidence: 0.8139,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_013' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.8293,
  latency: 210,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 3746,
  confidence: 0.4248,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_014' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_015' })
MERGE (a)-[r_014:TRIGGERS {
  strength: 0.8623,
  latency: 200,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 2659,
  confidence: 0.5081,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_015' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.8237,
  latency: 177,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6862,
  confidence: 0.2841,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_016' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.2638,
  latency: 187,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 3725,
  confidence: 0.8994,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_017' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.0783,
  latency: 161,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 1218,
  confidence: 0.0876,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_018' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.3222,
  latency: 112,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 6706,
  confidence: 0.4566,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_019' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_020' })
MERGE (a)-[r_019:PRODUCES {
  strength: 0.0299,
  latency: 125,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 188,
  confidence: 0.2075,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_020' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.1018,
  latency: 148,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 7735,
  confidence: 0.8085,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_021' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.3659,
  latency: 69,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 5420,
  confidence: 0.745,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_022' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.5442,
  latency: 157,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 7263,
  confidence: 0.6909,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_023' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.9412,
  latency: 200,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 6332,
  confidence: 0.6903,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_024' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.9428,
  latency: 88,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 3059,
  confidence: 0.5206,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_025' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.5316,
  latency: 241,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 5052,
  confidence: 0.491,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_026' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.4558,
  latency: 191,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 2281,
  confidence: 0.4923,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_027' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.3908,
  latency: 161,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 120,
  confidence: 0.4982,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_028' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_029' })
MERGE (a)-[r_028:ROUTES_TO {
  strength: 0.7063,
  latency: 39,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8752,
  confidence: 0.0624,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_04_registry_systems_2_029' }),
      (b:Compression { identifier: 'compression_04_registry_systems_2_000' })
MERGE (a)-[r_029:VALIDATES {
  strength: 0.6626,
  latency: 70,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 7850,
  confidence: 0.0619,
  active: true
}]->(b);
