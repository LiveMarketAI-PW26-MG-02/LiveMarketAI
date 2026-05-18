:param namespace => 'serializer_02_02';
:param batchSize => 32;
:param threshold => 0.681;
:param maxDepth => 5;
:param timeoutSeconds => 83;
:param region => 'us-west';
:param epoch => 84;
:param version => '4.4.9';

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_000' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.9012,
  latency: 54,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 5591,
  confidence: 0.8921,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_001' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.1873,
  latency: 6,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 1925,
  confidence: 0.9167,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_002' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.585,
  latency: 117,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 3194,
  confidence: 0.4035,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_003' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.2826,
  latency: 73,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 7684,
  confidence: 0.8637,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_004' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.7844,
  latency: 96,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 4733,
  confidence: 0.053,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_005' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.0358,
  latency: 199,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 8907,
  confidence: 0.4513,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_006' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.7589,
  latency: 249,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 555,
  confidence: 0.4732,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_007' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.7408,
  latency: 108,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 6423,
  confidence: 0.9536,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_008' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.721,
  latency: 29,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 3356,
  confidence: 0.001,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_009' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:TRIGGERS {
  strength: 0.6406,
  latency: 97,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 8431,
  confidence: 0.4306,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_010' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.2043,
  latency: 2,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9258,
  confidence: 0.7752,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_011' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:DEPENDS_ON {
  strength: 0.4152,
  latency: 127,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 4961,
  confidence: 0.6351,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_012' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.7047,
  latency: 96,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 6616,
  confidence: 0.4146,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_013' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.6431,
  latency: 52,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 7868,
  confidence: 0.1151,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_014' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:TRIGGERS {
  strength: 0.0313,
  latency: 141,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 6054,
  confidence: 0.0547,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_015' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.6621,
  latency: 164,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 910,
  confidence: 0.0105,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_016' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.4289,
  latency: 184,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 9881,
  confidence: 0.3879,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_017' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.8213,
  latency: 232,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 591,
  confidence: 0.0762,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_018' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.6166,
  latency: 32,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 2008,
  confidence: 0.4473,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_019' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.6833,
  latency: 51,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 148,
  confidence: 0.6921,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_020' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.1339,
  latency: 64,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 5975,
  confidence: 0.6424,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_021' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.132,
  latency: 140,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 1325,
  confidence: 0.5441,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_022' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.1234,
  latency: 152,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 2999,
  confidence: 0.172,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_023' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.4296,
  latency: 246,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 3951,
  confidence: 0.1483,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_024' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.2967,
  latency: 196,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 6575,
  confidence: 0.0219,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_025' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.822,
  latency: 38,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 3791,
  confidence: 0.5477,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_026' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.6568,
  latency: 124,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 3407,
  confidence: 0.911,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_027' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.163,
  latency: 168,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 3079,
  confidence: 0.4195,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_028' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.3595,
  latency: 198,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 1092,
  confidence: 0.7912,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_09_event_dispatchers_2_029' }),
      (b:Serializer { identifier: 'serializer_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.6343,
  latency: 223,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 6632,
  confidence: 0.7731,
  active: true
}]->(b);
