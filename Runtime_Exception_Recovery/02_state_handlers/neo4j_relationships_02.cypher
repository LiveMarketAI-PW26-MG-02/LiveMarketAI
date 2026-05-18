:param namespace => 'exceptionrecovery_02_02';
:param batchSize => 32;
:param threshold => 0.408;
:param maxDepth => 6;
:param timeoutSeconds => 73;
:param region => 'eu-west';
:param epoch => 40;
:param version => '4.3.0';

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_000' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.2123,
  latency: 20,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5730,
  confidence: 0.1294,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_001' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.9334,
  latency: 230,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 1959,
  confidence: 0.7504,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_002' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.2518,
  latency: 158,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 5181,
  confidence: 0.157,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_003' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.6595,
  latency: 75,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 8126,
  confidence: 0.7761,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_004' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.6726,
  latency: 233,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 730,
  confidence: 0.0701,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_005' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.9367,
  latency: 152,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 1857,
  confidence: 0.7011,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_006' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.2901,
  latency: 10,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 837,
  confidence: 0.5652,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_007' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.0266,
  latency: 170,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 2819,
  confidence: 0.0121,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_008' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.6177,
  latency: 211,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 3736,
  confidence: 0.8241,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_009' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.1302,
  latency: 20,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 5798,
  confidence: 0.7523,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_010' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.3719,
  latency: 82,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1762,
  confidence: 0.3136,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_011' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.9892,
  latency: 46,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 9453,
  confidence: 0.9022,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_012' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.823,
  latency: 172,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 9512,
  confidence: 0.7292,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_013' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.0538,
  latency: 131,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 8830,
  confidence: 0.3551,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_014' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.5142,
  latency: 222,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 2211,
  confidence: 0.0119,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_015' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.221,
  latency: 17,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 6015,
  confidence: 0.6163,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_016' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.2018,
  latency: 27,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 2363,
  confidence: 0.0811,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_017' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.6046,
  latency: 152,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 1488,
  confidence: 0.7596,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_018' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.2517,
  latency: 38,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 1389,
  confidence: 0.3971,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_019' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.9164,
  latency: 89,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 2047,
  confidence: 0.5402,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_020' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.9812,
  latency: 218,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 5881,
  confidence: 0.6322,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_021' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.5609,
  latency: 6,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 7042,
  confidence: 0.9291,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_022' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.1342,
  latency: 199,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 8345,
  confidence: 0.4525,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_023' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.346,
  latency: 38,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 8091,
  confidence: 0.1736,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_024' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.4851,
  latency: 10,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 2998,
  confidence: 0.1599,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_025' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.3334,
  latency: 214,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 4885,
  confidence: 0.8415,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_026' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_027' })
MERGE (a)-[r_026:CALIBRATES {
  strength: 0.9373,
  latency: 172,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 247,
  confidence: 0.2988,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_027' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.6381,
  latency: 168,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 4250,
  confidence: 0.643,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_028' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.434,
  latency: 79,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 8815,
  confidence: 0.5062,
  active: true
}]->(b);

MATCH (a:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_029' }),
      (b:ExceptionRecovery { identifier: 'exceptionrecovery_02_state_handlers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.381,
  latency: 202,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 5011,
  confidence: 0.1313,
  active: true
}]->(b);
