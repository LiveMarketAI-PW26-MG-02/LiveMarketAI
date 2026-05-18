:param namespace => 'explainability_02_02';
:param batchSize => 256;
:param threshold => 0.779;
:param maxDepth => 9;
:param timeoutSeconds => 67;
:param region => 'us-east';
:param epoch => 7;
:param version => '1.6.2';

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_000' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.0209,
  latency: 203,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 6558,
  confidence: 0.2725,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_001' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.2208,
  latency: 30,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 7601,
  confidence: 0.64,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_002' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.8512,
  latency: 87,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 8520,
  confidence: 0.6314,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_003' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_004' })
MERGE (a)-[r_003:OBSERVES {
  strength: 0.3657,
  latency: 155,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 2587,
  confidence: 0.1989,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_004' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.5068,
  latency: 179,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 9770,
  confidence: 0.0341,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_005' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.0992,
  latency: 34,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 2323,
  confidence: 0.9331,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_006' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.1394,
  latency: 66,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 5991,
  confidence: 0.256,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_007' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.5302,
  latency: 24,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 4552,
  confidence: 0.3055,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_008' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.982,
  latency: 181,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 7825,
  confidence: 0.8205,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_009' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.77,
  latency: 247,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 9767,
  confidence: 0.8607,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_010' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.5359,
  latency: 234,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 6431,
  confidence: 0.222,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_011' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.5683,
  latency: 32,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 4058,
  confidence: 0.7194,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_012' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.2277,
  latency: 18,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 7056,
  confidence: 0.0521,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_013' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_014' })
MERGE (a)-[r_013:MONITORS {
  strength: 0.07,
  latency: 2,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 5336,
  confidence: 0.8713,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_014' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.12,
  latency: 104,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 8640,
  confidence: 0.0389,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_015' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.1408,
  latency: 11,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 5717,
  confidence: 0.9212,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_016' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.4651,
  latency: 136,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 567,
  confidence: 0.2767,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_017' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.7716,
  latency: 14,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 3398,
  confidence: 0.2126,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_018' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.8892,
  latency: 100,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 8207,
  confidence: 0.8226,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_019' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.8228,
  latency: 76,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 8658,
  confidence: 0.744,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_020' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.1864,
  latency: 119,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 6207,
  confidence: 0.562,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_021' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.1434,
  latency: 192,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 245,
  confidence: 0.7592,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_022' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_023' })
MERGE (a)-[r_022:DEPENDS_ON {
  strength: 0.7682,
  latency: 218,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 2881,
  confidence: 0.3667,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_023' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.7081,
  latency: 206,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 3616,
  confidence: 0.6576,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_024' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.5227,
  latency: 129,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 460,
  confidence: 0.2121,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_025' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.0949,
  latency: 130,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 2562,
  confidence: 0.6848,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_026' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.2103,
  latency: 43,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 606,
  confidence: 0.9413,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_027' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.4299,
  latency: 232,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 1876,
  confidence: 0.8146,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_028' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.1695,
  latency: 195,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 3326,
  confidence: 0.7976,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_05_metric_trackers_2_029' }),
      (b:Explainability { identifier: 'explainability_05_metric_trackers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.7349,
  latency: 144,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 9899,
  confidence: 0.1866,
  active: true
}]->(b);
