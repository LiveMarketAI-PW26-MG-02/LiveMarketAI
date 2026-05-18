:param namespace => 'exceptionrecovery_02_02';
:param batchSize => 512;
:param threshold => 0.843;
:param maxDepth => 11;
:param timeoutSeconds => 90;
:param region => 'eu-west';
:param epoch => 73;
:param version => '1.8.6';

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_000' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_001' })
MERGE (a)-[r_000:PRODUCES {
  strength: 0.3629,
  latency: 171,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 7565,
  confidence: 0.8796,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_001' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.67,
  latency: 175,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 3900,
  confidence: 0.8307,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_002' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.8366,
  latency: 65,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 5209,
  confidence: 0.9318,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_003' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.8841,
  latency: 192,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 5729,
  confidence: 0.7291,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_004' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.0555,
  latency: 130,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 4238,
  confidence: 0.4612,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_005' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.5742,
  latency: 218,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 4622,
  confidence: 0.6109,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_006' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.2842,
  latency: 83,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 3795,
  confidence: 0.2363,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_007' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.3942,
  latency: 13,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 9030,
  confidence: 0.1835,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_008' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_009' })
MERGE (a)-[r_008:PRODUCES {
  strength: 0.3053,
  latency: 67,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5670,
  confidence: 0.4671,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_009' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.1943,
  latency: 124,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 6746,
  confidence: 0.4939,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_010' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.6021,
  latency: 188,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 8693,
  confidence: 0.3043,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_011' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.1342,
  latency: 113,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 9339,
  confidence: 0.1218,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_012' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.8221,
  latency: 204,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7745,
  confidence: 0.4758,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_013' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.7305,
  latency: 77,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 4068,
  confidence: 0.362,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_014' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.9709,
  latency: 156,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 2158,
  confidence: 0.0165,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_015' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.2959,
  latency: 122,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 9265,
  confidence: 0.2133,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_016' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.3322,
  latency: 143,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 666,
  confidence: 0.1873,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_017' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.4951,
  latency: 142,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1078,
  confidence: 0.2862,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_018' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.0007,
  latency: 136,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4089,
  confidence: 0.1726,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_019' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.226,
  latency: 176,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 3399,
  confidence: 0.484,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_020' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.5705,
  latency: 80,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6909,
  confidence: 0.021,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_021' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.5085,
  latency: 56,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 9519,
  confidence: 0.535,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_022' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.126,
  latency: 64,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 9195,
  confidence: 0.7641,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_023' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.7425,
  latency: 97,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 3769,
  confidence: 0.2901,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_024' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.0715,
  latency: 181,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 3145,
  confidence: 0.12,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_025' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.3537,
  latency: 105,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 3504,
  confidence: 0.1435,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_026' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.2056,
  latency: 41,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 8809,
  confidence: 0.7939,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_027' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.246,
  latency: 61,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 845,
  confidence: 0.1483,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_028' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.6602,
  latency: 130,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 9584,
  confidence: 0.6015,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_029' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_03_config_managers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.4949,
  latency: 135,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 8908,
  confidence: 0.86,
  active: true
}]->(b);
