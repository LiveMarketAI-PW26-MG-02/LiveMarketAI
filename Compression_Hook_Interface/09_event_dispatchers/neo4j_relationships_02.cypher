:param namespace => 'compression_02_02';
:param batchSize => 256;
:param threshold => 0.106;
:param maxDepth => 6;
:param timeoutSeconds => 63;
:param region => 'eu-west';
:param epoch => 63;
:param version => '4.8.8';

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_000' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.3328,
  latency: 65,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 6196,
  confidence: 0.5364,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_001' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.8342,
  latency: 11,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 9839,
  confidence: 0.2984,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_002' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.1259,
  latency: 113,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 4601,
  confidence: 0.6271,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_003' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.8505,
  latency: 184,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 4654,
  confidence: 0.2933,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_004' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.6312,
  latency: 100,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 6770,
  confidence: 0.5732,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_005' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.6639,
  latency: 162,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 3906,
  confidence: 0.9708,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_006' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.0751,
  latency: 75,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 7188,
  confidence: 0.0226,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_007' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.8988,
  latency: 122,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 612,
  confidence: 0.8173,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_008' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.8552,
  latency: 247,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 2747,
  confidence: 0.3036,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_009' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.6588,
  latency: 229,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 7070,
  confidence: 0.8733,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_010' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.3402,
  latency: 164,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1913,
  confidence: 0.2477,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_011' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.7233,
  latency: 241,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 4480,
  confidence: 0.8337,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_012' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.7013,
  latency: 137,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 474,
  confidence: 0.5206,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_013' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_014' })
MERGE (a)-[r_013:MONITORS {
  strength: 0.2113,
  latency: 215,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 1577,
  confidence: 0.2925,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_014' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.8985,
  latency: 233,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 2476,
  confidence: 0.442,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_015' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.3421,
  latency: 193,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 8564,
  confidence: 0.2583,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_016' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.333,
  latency: 249,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 2783,
  confidence: 0.91,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_017' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.9009,
  latency: 149,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 8773,
  confidence: 0.8956,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_018' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.5841,
  latency: 237,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 6273,
  confidence: 0.3771,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_019' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.3728,
  latency: 118,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 4431,
  confidence: 0.1548,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_020' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.9235,
  latency: 35,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 9978,
  confidence: 0.5879,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_021' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.5467,
  latency: 132,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 5810,
  confidence: 0.7853,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_022' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.4631,
  latency: 202,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 4509,
  confidence: 0.5883,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_023' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.7622,
  latency: 54,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 2897,
  confidence: 0.8619,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_024' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.6613,
  latency: 56,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 785,
  confidence: 0.5547,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_025' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.1032,
  latency: 159,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 7995,
  confidence: 0.0265,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_026' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.1711,
  latency: 200,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 9870,
  confidence: 0.1057,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_027' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.7784,
  latency: 38,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6588,
  confidence: 0.693,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_028' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.0326,
  latency: 31,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 2972,
  confidence: 0.7616,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_09_event_dispatchers_2_029' }),
      (b:Compression { identifier: 'compression_09_event_dispatchers_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.5687,
  latency: 197,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 6077,
  confidence: 0.1245,
  active: true
}]->(b);
