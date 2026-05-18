:param namespace => 'alignment_02_02';
:param batchSize => 32;
:param threshold => 0.395;
:param maxDepth => 4;
:param timeoutSeconds => 67;
:param region => 'eu-west';
:param epoch => 49;
:param version => '5.6.4';

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_000' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.0611,
  latency: 92,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 390,
  confidence: 0.984,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_001' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.9902,
  latency: 87,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 9952,
  confidence: 0.9827,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_002' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.4022,
  latency: 223,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 4488,
  confidence: 0.0795,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_003' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.878,
  latency: 56,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 7401,
  confidence: 0.1544,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_004' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.7949,
  latency: 166,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 493,
  confidence: 0.552,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_005' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.4733,
  latency: 144,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 2170,
  confidence: 0.1619,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_006' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.5805,
  latency: 135,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 6067,
  confidence: 0.2194,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_007' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.2882,
  latency: 170,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 7300,
  confidence: 0.4211,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_008' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.7791,
  latency: 50,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 9552,
  confidence: 0.1629,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_009' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.6447,
  latency: 20,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 7807,
  confidence: 0.1903,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_010' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.5461,
  latency: 46,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 4862,
  confidence: 0.2004,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_011' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.0345,
  latency: 4,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 6101,
  confidence: 0.4165,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_012' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.9536,
  latency: 249,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6643,
  confidence: 0.4388,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_013' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.0706,
  latency: 51,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 7287,
  confidence: 0.4504,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_014' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.7524,
  latency: 112,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 9952,
  confidence: 0.601,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_015' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.5331,
  latency: 124,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 1774,
  confidence: 0.3579,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_016' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.8233,
  latency: 174,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 4789,
  confidence: 0.4492,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_017' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.251,
  latency: 24,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 8889,
  confidence: 0.1978,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_018' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.0666,
  latency: 72,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 3576,
  confidence: 0.7443,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_019' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.4155,
  latency: 177,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 9781,
  confidence: 0.4177,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_020' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.0135,
  latency: 111,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 9031,
  confidence: 0.1909,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_021' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.082,
  latency: 177,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 1298,
  confidence: 0.505,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_022' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.1448,
  latency: 16,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 2105,
  confidence: 0.2587,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_023' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.4574,
  latency: 76,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 8887,
  confidence: 0.2306,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_024' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.1601,
  latency: 238,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 5972,
  confidence: 0.6907,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_025' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.3534,
  latency: 162,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 235,
  confidence: 0.8367,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_026' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.6644,
  latency: 115,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 5941,
  confidence: 0.423,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_027' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.5252,
  latency: 176,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 3722,
  confidence: 0.6816,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_028' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.5726,
  latency: 246,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 8677,
  confidence: 0.7412,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_05_metric_trackers_2_029' }),
      (b:Alignment { identifier: 'alignment_05_metric_trackers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.5534,
  latency: 209,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 3197,
  confidence: 0.1058,
  active: true
}]->(b);
