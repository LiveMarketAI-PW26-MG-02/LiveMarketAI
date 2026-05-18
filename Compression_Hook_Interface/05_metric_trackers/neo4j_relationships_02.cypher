:param namespace => 'compression_02_02';
:param batchSize => 64;
:param threshold => 0.855;
:param maxDepth => 8;
:param timeoutSeconds => 101;
:param region => 'us-west';
:param epoch => 98;
:param version => '3.1.1';

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_000' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.9647,
  latency: 51,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 8863,
  confidence: 0.1706,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_001' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.8908,
  latency: 165,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 4006,
  confidence: 0.8024,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_002' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.1759,
  latency: 96,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 7864,
  confidence: 0.3262,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_003' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_004' })
MERGE (a)-[r_003:OBSERVES {
  strength: 0.7146,
  latency: 109,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 9160,
  confidence: 0.0092,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_004' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.5886,
  latency: 205,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 2370,
  confidence: 0.5371,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_005' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.7605,
  latency: 9,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 9247,
  confidence: 0.8445,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_006' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.7304,
  latency: 65,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 2061,
  confidence: 0.2372,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_007' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.4139,
  latency: 137,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 4574,
  confidence: 0.5385,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_008' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.6739,
  latency: 59,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 6349,
  confidence: 0.9709,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_009' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_010' })
MERGE (a)-[r_009:PRODUCES {
  strength: 0.8721,
  latency: 67,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 5975,
  confidence: 0.9295,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_010' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.1962,
  latency: 217,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 551,
  confidence: 0.8529,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_011' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.3207,
  latency: 129,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 9515,
  confidence: 0.94,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_012' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.5312,
  latency: 210,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 4105,
  confidence: 0.003,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_013' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.4608,
  latency: 203,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 1059,
  confidence: 0.127,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_014' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.1678,
  latency: 156,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 9805,
  confidence: 0.9154,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_015' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.3661,
  latency: 228,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 3031,
  confidence: 0.1309,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_016' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.9766,
  latency: 73,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9490,
  confidence: 0.4669,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_017' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.6529,
  latency: 48,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 1094,
  confidence: 0.4581,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_018' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.1285,
  latency: 216,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 8062,
  confidence: 0.126,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_019' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.3299,
  latency: 182,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 7125,
  confidence: 0.0189,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_020' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.6462,
  latency: 17,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 9160,
  confidence: 0.2697,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_021' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.2927,
  latency: 33,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 5549,
  confidence: 0.676,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_022' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.615,
  latency: 71,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 9340,
  confidence: 0.967,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_023' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.819,
  latency: 41,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 2627,
  confidence: 0.1777,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_024' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.458,
  latency: 243,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 2875,
  confidence: 0.7518,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_025' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.5842,
  latency: 128,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 9077,
  confidence: 0.9494,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_026' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.3552,
  latency: 194,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 923,
  confidence: 0.1403,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_027' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_028' })
MERGE (a)-[r_027:CALIBRATES {
  strength: 0.4282,
  latency: 115,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 9518,
  confidence: 0.8666,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_028' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.4137,
  latency: 133,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 9815,
  confidence: 0.9311,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_05_metric_trackers_2_029' }),
      (b:Compression { identifier: 'compression_05_metric_trackers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.2422,
  latency: 221,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 5472,
  confidence: 0.8278,
  active: true
}]->(b);
