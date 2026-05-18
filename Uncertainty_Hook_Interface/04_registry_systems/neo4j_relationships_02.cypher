:param namespace => 'uncertainty_02_02';
:param batchSize => 32;
:param threshold => 0.686;
:param maxDepth => 5;
:param timeoutSeconds => 79;
:param region => 'us-west';
:param epoch => 61;
:param version => '5.6.8';

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_000' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.2704,
  latency: 234,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 8940,
  confidence: 0.8744,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_001' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.4857,
  latency: 238,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 5343,
  confidence: 0.3953,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_002' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.5246,
  latency: 147,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 2236,
  confidence: 0.4933,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_003' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.2301,
  latency: 178,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 7749,
  confidence: 0.9199,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_004' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.1268,
  latency: 15,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 6212,
  confidence: 0.2186,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_005' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.2325,
  latency: 124,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 643,
  confidence: 0.8758,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_006' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.0445,
  latency: 215,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 7542,
  confidence: 0.0252,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_007' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.4396,
  latency: 204,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 5421,
  confidence: 0.9576,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_008' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.2123,
  latency: 61,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 5244,
  confidence: 0.2804,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_009' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.7803,
  latency: 197,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6292,
  confidence: 0.0842,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_010' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.6315,
  latency: 140,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 3016,
  confidence: 0.2523,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_011' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.8786,
  latency: 180,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 1648,
  confidence: 0.1219,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_012' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.3644,
  latency: 211,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 2522,
  confidence: 0.0549,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_013' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.1522,
  latency: 73,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 3276,
  confidence: 0.867,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_014' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.7056,
  latency: 110,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 7956,
  confidence: 0.0997,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_015' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.6289,
  latency: 144,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 7831,
  confidence: 0.7063,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_016' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.9139,
  latency: 185,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 7260,
  confidence: 0.2226,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_017' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.1117,
  latency: 27,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 7144,
  confidence: 0.2682,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_018' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.093,
  latency: 44,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 3062,
  confidence: 0.1003,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_019' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.0281,
  latency: 2,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 3245,
  confidence: 0.0742,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_020' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.2001,
  latency: 14,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8471,
  confidence: 0.9573,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_021' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.123,
  latency: 165,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 2752,
  confidence: 0.7527,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_022' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.5532,
  latency: 17,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 8602,
  confidence: 0.5224,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_023' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_024' })
MERGE (a)-[r_023:MONITORS {
  strength: 0.5485,
  latency: 113,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 5853,
  confidence: 0.0119,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_024' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.5076,
  latency: 62,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 4666,
  confidence: 0.108,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_025' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.4052,
  latency: 182,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 6079,
  confidence: 0.2808,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_026' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.1383,
  latency: 215,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 5587,
  confidence: 0.2433,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_027' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.52,
  latency: 54,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 5388,
  confidence: 0.5493,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_028' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.2787,
  latency: 118,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 4547,
  confidence: 0.3283,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_029' }),
      (b:Uncertainty { identifier: 'uncertainty_04_registry_systems_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.5037,
  latency: 159,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 1248,
  confidence: 0.4684,
  active: true
}]->(b);
