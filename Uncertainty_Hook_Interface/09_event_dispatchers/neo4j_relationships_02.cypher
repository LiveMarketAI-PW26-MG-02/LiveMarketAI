:param namespace => 'uncertainty_02_02';
:param batchSize => 64;
:param threshold => 0.234;
:param maxDepth => 9;
:param timeoutSeconds => 77;
:param region => 'ap-south';
:param epoch => 60;
:param version => '1.2.3';

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_000' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:PRODUCES {
  strength: 0.5597,
  latency: 66,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 2592,
  confidence: 0.0228,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_001' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.8561,
  latency: 54,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 1384,
  confidence: 0.8683,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_002' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.1895,
  latency: 9,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 6918,
  confidence: 0.611,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_003' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.6257,
  latency: 184,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 8465,
  confidence: 0.0053,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_004' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.41,
  latency: 35,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 9608,
  confidence: 0.8485,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_005' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.1379,
  latency: 114,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 4710,
  confidence: 0.4842,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_006' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.9534,
  latency: 185,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 7282,
  confidence: 0.2676,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_007' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.8036,
  latency: 56,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 179,
  confidence: 0.0116,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_008' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.3478,
  latency: 64,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 402,
  confidence: 0.5655,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_009' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.9089,
  latency: 169,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 4344,
  confidence: 0.8078,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_010' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.9181,
  latency: 70,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1302,
  confidence: 0.7843,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_011' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.3804,
  latency: 99,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 8531,
  confidence: 0.161,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_012' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.2449,
  latency: 122,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 6270,
  confidence: 0.6959,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_013' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.264,
  latency: 146,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 194,
  confidence: 0.2118,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_014' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.1283,
  latency: 126,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 4967,
  confidence: 0.8154,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_015' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.6443,
  latency: 154,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 1051,
  confidence: 0.1096,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_016' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.9945,
  latency: 244,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 1048,
  confidence: 0.6875,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_017' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:PRODUCES {
  strength: 0.6189,
  latency: 85,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 2680,
  confidence: 0.1349,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_018' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:CALIBRATES {
  strength: 0.5348,
  latency: 193,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 2792,
  confidence: 0.6035,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_019' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.2014,
  latency: 157,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 325,
  confidence: 0.2517,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_020' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.0378,
  latency: 82,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 3527,
  confidence: 0.6632,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_021' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.4942,
  latency: 95,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 7431,
  confidence: 0.3961,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_022' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.4091,
  latency: 84,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 9261,
  confidence: 0.2387,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_023' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.3523,
  latency: 215,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 2698,
  confidence: 0.9544,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_024' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.3147,
  latency: 244,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 3728,
  confidence: 0.3515,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_025' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.7927,
  latency: 72,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 6679,
  confidence: 0.8358,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_026' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.658,
  latency: 26,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 1443,
  confidence: 0.4005,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_027' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.0349,
  latency: 106,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 2628,
  confidence: 0.8368,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_028' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:ROUTES_TO {
  strength: 0.8222,
  latency: 59,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 6120,
  confidence: 0.846,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_029' }),
      (b:Uncertainty { identifier: 'uncertainty_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.2361,
  latency: 10,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 1050,
  confidence: 0.262,
  active: true
}]->(b);
