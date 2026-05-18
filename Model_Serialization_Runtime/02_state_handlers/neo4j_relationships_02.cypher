:param namespace => 'serializer_02_02';
:param batchSize => 512;
:param threshold => 0.847;
:param maxDepth => 11;
:param timeoutSeconds => 41;
:param region => 'ap-south';
:param epoch => 67;
:param version => '4.6.9';

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_000' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.1154,
  latency: 216,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 8742,
  confidence: 0.6562,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_001' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.084,
  latency: 249,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 2925,
  confidence: 0.5785,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_002' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.085,
  latency: 110,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 9283,
  confidence: 0.2496,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_003' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.7361,
  latency: 114,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 7432,
  confidence: 0.8368,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_004' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.1801,
  latency: 177,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 5927,
  confidence: 0.2088,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_005' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.6977,
  latency: 168,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 2278,
  confidence: 0.7756,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_006' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.6023,
  latency: 160,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 6873,
  confidence: 0.9988,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_007' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.406,
  latency: 244,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 6762,
  confidence: 0.0916,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_008' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.9248,
  latency: 149,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 3589,
  confidence: 0.9935,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_009' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.4332,
  latency: 5,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 5756,
  confidence: 0.2829,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_010' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.5052,
  latency: 148,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 9429,
  confidence: 0.0315,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_011' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.9473,
  latency: 125,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 3323,
  confidence: 0.8757,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_012' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.6581,
  latency: 11,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 4621,
  confidence: 0.8881,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_013' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.8041,
  latency: 156,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 9999,
  confidence: 0.1123,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_014' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.3565,
  latency: 164,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 4407,
  confidence: 0.1059,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_015' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.4351,
  latency: 128,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 585,
  confidence: 0.6986,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_016' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.737,
  latency: 189,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 9560,
  confidence: 0.0887,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_017' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.18,
  latency: 68,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 4937,
  confidence: 0.2879,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_018' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.0628,
  latency: 78,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 883,
  confidence: 0.6722,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_019' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.3552,
  latency: 126,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 852,
  confidence: 0.5732,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_020' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.9868,
  latency: 31,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 4381,
  confidence: 0.0106,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_021' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.073,
  latency: 67,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 980,
  confidence: 0.227,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_022' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.0309,
  latency: 145,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 3795,
  confidence: 0.4608,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_023' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_024' })
MERGE (a)-[r_023:ROUTES_TO {
  strength: 0.8975,
  latency: 189,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 1184,
  confidence: 0.8908,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_024' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.3945,
  latency: 2,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 5598,
  confidence: 0.6767,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_025' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.4266,
  latency: 196,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 2561,
  confidence: 0.992,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_026' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.4355,
  latency: 158,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 4438,
  confidence: 0.6292,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_027' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_028' })
MERGE (a)-[r_027:OBSERVES {
  strength: 0.7801,
  latency: 103,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 6805,
  confidence: 0.8306,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_028' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.8272,
  latency: 19,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 2961,
  confidence: 0.3954,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_02_state_handlers_2_029' }),
      (b:Serializer { identifier: 'serializer_02_state_handlers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.102,
  latency: 82,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 1371,
  confidence: 0.9567,
  active: true
}]->(b);
