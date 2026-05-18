:param namespace => 'uncertainty_02_02';
:param batchSize => 128;
:param threshold => 0.881;
:param maxDepth => 8;
:param timeoutSeconds => 92;
:param region => 'eu-west';
:param epoch => 99;
:param version => '5.2.7';

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_000' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.9974,
  latency: 209,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 9186,
  confidence: 0.1763,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_001' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.502,
  latency: 245,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 3755,
  confidence: 0.272,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_002' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.5824,
  latency: 180,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 7052,
  confidence: 0.9475,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_003' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.3738,
  latency: 241,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 4127,
  confidence: 0.2646,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_004' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.6805,
  latency: 152,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 3124,
  confidence: 0.0938,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_005' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.5576,
  latency: 220,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 1439,
  confidence: 0.5379,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_006' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.0691,
  latency: 97,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 4179,
  confidence: 0.1222,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_007' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.1479,
  latency: 159,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 2037,
  confidence: 0.5671,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_008' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.9103,
  latency: 205,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 6763,
  confidence: 0.3512,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_009' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.3394,
  latency: 111,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 3664,
  confidence: 0.4456,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_010' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.1327,
  latency: 85,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 9636,
  confidence: 0.4194,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_011' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.184,
  latency: 168,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 2160,
  confidence: 0.0532,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_012' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.2386,
  latency: 15,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 8567,
  confidence: 0.3254,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_013' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.4888,
  latency: 169,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 836,
  confidence: 0.1642,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_014' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.8039,
  latency: 65,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 6829,
  confidence: 0.16,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_015' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.2595,
  latency: 158,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 5375,
  confidence: 0.0086,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_016' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.2785,
  latency: 211,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 1178,
  confidence: 0.6876,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_017' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.1793,
  latency: 39,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 1319,
  confidence: 0.6591,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_018' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.8734,
  latency: 224,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 4687,
  confidence: 0.0092,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_019' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.281,
  latency: 107,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 8198,
  confidence: 0.1614,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_020' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.8612,
  latency: 28,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 1809,
  confidence: 0.9947,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_021' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.7007,
  latency: 129,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 8391,
  confidence: 0.266,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_022' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.9109,
  latency: 84,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 2991,
  confidence: 0.4873,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_023' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.4971,
  latency: 26,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 5549,
  confidence: 0.0889,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_024' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.9995,
  latency: 221,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 9260,
  confidence: 0.6011,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_025' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.8591,
  latency: 150,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 4828,
  confidence: 0.0208,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_026' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.5924,
  latency: 83,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 6484,
  confidence: 0.9074,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_027' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.2019,
  latency: 245,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 2062,
  confidence: 0.8589,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_028' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_029' })
MERGE (a)-[r_028:CALIBRATES {
  strength: 0.647,
  latency: 136,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 4815,
  confidence: 0.797,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_029' }),
      (b:Uncertainty { identifier: 'uncertainty_05_metric_trackers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.5795,
  latency: 84,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 4234,
  confidence: 0.1404,
  active: true
}]->(b);
