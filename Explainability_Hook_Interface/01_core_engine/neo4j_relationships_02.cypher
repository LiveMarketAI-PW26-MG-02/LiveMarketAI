:param namespace => 'explainability_02_02';
:param batchSize => 32;
:param threshold => 0.105;
:param maxDepth => 7;
:param timeoutSeconds => 52;
:param region => 'us-east';
:param epoch => 64;
:param version => '3.6.2';

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_000' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.4212,
  latency: 163,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9491,
  confidence: 0.8227,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_001' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.7082,
  latency: 91,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 999,
  confidence: 0.6477,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_002' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.1404,
  latency: 16,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 4746,
  confidence: 0.3573,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_003' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.0795,
  latency: 101,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 8478,
  confidence: 0.5972,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_004' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.2567,
  latency: 221,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 1522,
  confidence: 0.6636,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_005' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.0702,
  latency: 144,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 146,
  confidence: 0.2034,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_006' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.6967,
  latency: 134,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 8194,
  confidence: 0.4337,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_007' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.5219,
  latency: 225,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 780,
  confidence: 0.9031,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_008' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.1326,
  latency: 44,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 3073,
  confidence: 0.0871,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_009' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.2397,
  latency: 27,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 2235,
  confidence: 0.3614,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_010' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.0888,
  latency: 89,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 7215,
  confidence: 0.1531,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_011' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.9632,
  latency: 144,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 2715,
  confidence: 0.5813,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_012' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.958,
  latency: 140,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 967,
  confidence: 0.7893,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_013' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.0732,
  latency: 114,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 8424,
  confidence: 0.5456,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_014' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.3107,
  latency: 122,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 7426,
  confidence: 0.0824,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_015' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.0264,
  latency: 11,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 2945,
  confidence: 0.4596,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_016' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.1751,
  latency: 125,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 7549,
  confidence: 0.3601,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_017' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.8888,
  latency: 213,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 5772,
  confidence: 0.0784,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_018' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.5569,
  latency: 137,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 5582,
  confidence: 0.777,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_019' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.4297,
  latency: 216,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 5058,
  confidence: 0.5658,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_020' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.228,
  latency: 18,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 8926,
  confidence: 0.0898,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_021' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.7111,
  latency: 32,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 4753,
  confidence: 0.8476,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_022' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.687,
  latency: 38,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 6296,
  confidence: 0.6157,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_023' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.8861,
  latency: 189,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 1835,
  confidence: 0.5388,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_024' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.1383,
  latency: 151,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 4831,
  confidence: 0.0119,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_025' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.2559,
  latency: 239,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 3206,
  confidence: 0.5658,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_026' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.6956,
  latency: 97,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 6546,
  confidence: 0.2774,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_027' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.1648,
  latency: 67,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1982,
  confidence: 0.4482,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_028' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.9229,
  latency: 98,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 226,
  confidence: 0.0782,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_01_core_engine_2_029' }),
      (b:Explainability { identifier: 'explainability_01_core_engine_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.3808,
  latency: 37,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 9044,
  confidence: 0.8584,
  active: true
}]->(b);
