:param namespace => 'batchinference_02_02';
:param batchSize => 32;
:param threshold => 0.382;
:param maxDepth => 11;
:param timeoutSeconds => 118;
:param region => 'us-west';
:param epoch => 51;
:param version => '5.2.2';

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_000' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.6451,
  latency: 78,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 7612,
  confidence: 0.0153,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_001' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_002' })
MERGE (a)-[r_001:VALIDATES {
  strength: 0.0693,
  latency: 221,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 7367,
  confidence: 0.3706,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_002' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.5294,
  latency: 111,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6958,
  confidence: 0.1609,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_003' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.5657,
  latency: 113,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 3212,
  confidence: 0.4812,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_004' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.8319,
  latency: 139,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 7439,
  confidence: 0.645,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_005' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.3366,
  latency: 219,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 2774,
  confidence: 0.3073,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_006' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.0977,
  latency: 128,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 6113,
  confidence: 0.7686,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_007' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.8141,
  latency: 194,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 6361,
  confidence: 0.9946,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_008' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.0977,
  latency: 165,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 1125,
  confidence: 0.6285,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_009' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_010' })
MERGE (a)-[r_009:PRODUCES {
  strength: 0.2347,
  latency: 136,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 148,
  confidence: 0.9193,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_010' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.6911,
  latency: 176,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 6194,
  confidence: 0.9928,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_011' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.4145,
  latency: 73,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 8634,
  confidence: 0.2049,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_012' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.2278,
  latency: 25,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 917,
  confidence: 0.9086,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_013' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_014' })
MERGE (a)-[r_013:MONITORS {
  strength: 0.3773,
  latency: 178,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 7942,
  confidence: 0.6577,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_014' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.6782,
  latency: 122,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 4662,
  confidence: 0.4457,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_015' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.1749,
  latency: 6,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 6072,
  confidence: 0.4635,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_016' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.5177,
  latency: 67,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 4833,
  confidence: 0.8485,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_017' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.2174,
  latency: 203,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 3803,
  confidence: 0.7501,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_018' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.3236,
  latency: 82,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 6743,
  confidence: 0.8034,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_019' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.0171,
  latency: 152,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 7207,
  confidence: 0.2967,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_020' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.6598,
  latency: 182,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 1872,
  confidence: 0.3797,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_021' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.8812,
  latency: 47,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 377,
  confidence: 0.7741,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_022' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.1818,
  latency: 178,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 5646,
  confidence: 0.9542,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_023' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.6759,
  latency: 209,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 6371,
  confidence: 0.2115,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_024' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.4168,
  latency: 45,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 7222,
  confidence: 0.7361,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_025' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.1774,
  latency: 136,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 1729,
  confidence: 0.0279,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_026' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.6702,
  latency: 240,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9643,
  confidence: 0.4199,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_027' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_028' })
MERGE (a)-[r_027:CALIBRATES {
  strength: 0.5337,
  latency: 107,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 9139,
  confidence: 0.7397,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_028' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.544,
  latency: 238,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 1008,
  confidence: 0.8593,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_05_metric_trackers_2_029' }),
      (b:BatchInference { identifier: 'batchinference_05_metric_trackers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.6753,
  latency: 113,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 1573,
  confidence: 0.0681,
  active: true
}]->(b);
