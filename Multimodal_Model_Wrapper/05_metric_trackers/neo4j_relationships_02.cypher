:param namespace => 'multimodal_02_02';
:param batchSize => 512;
:param threshold => 0.448;
:param maxDepth => 8;
:param timeoutSeconds => 111;
:param region => 'ap-south';
:param epoch => 91;
:param version => '4.8.5';

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_000' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.6798,
  latency: 194,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 506,
  confidence: 0.884,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_001' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.3656,
  latency: 65,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 6272,
  confidence: 0.4871,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_002' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.4256,
  latency: 180,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 4403,
  confidence: 0.792,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_003' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_004' })
MERGE (a)-[r_003:ROUTES_TO {
  strength: 0.1425,
  latency: 201,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5296,
  confidence: 0.9319,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_004' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.8656,
  latency: 62,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 430,
  confidence: 0.382,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_005' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_006' })
MERGE (a)-[r_005:MONITORS {
  strength: 0.6573,
  latency: 158,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 2278,
  confidence: 0.6313,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_006' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.1613,
  latency: 154,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 6908,
  confidence: 0.84,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_007' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.806,
  latency: 20,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 8013,
  confidence: 0.6799,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_008' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.9011,
  latency: 191,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 3798,
  confidence: 0.3982,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_009' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.009,
  latency: 109,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 9965,
  confidence: 0.8437,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_010' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.0596,
  latency: 163,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 240,
  confidence: 0.1246,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_011' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.5196,
  latency: 31,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 9647,
  confidence: 0.4342,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_012' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.075,
  latency: 122,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 8730,
  confidence: 0.9116,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_013' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.0641,
  latency: 15,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 7582,
  confidence: 0.2284,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_014' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.8244,
  latency: 79,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 6932,
  confidence: 0.6721,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_015' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.0967,
  latency: 16,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 3226,
  confidence: 0.6168,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_016' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.633,
  latency: 77,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 5343,
  confidence: 0.3818,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_017' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.0532,
  latency: 145,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1837,
  confidence: 0.2234,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_018' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.5918,
  latency: 248,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 8707,
  confidence: 0.5362,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_019' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.008,
  latency: 177,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 1240,
  confidence: 0.1156,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_020' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.8511,
  latency: 44,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7572,
  confidence: 0.5474,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_021' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.8883,
  latency: 69,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 2189,
  confidence: 0.8305,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_022' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.6876,
  latency: 11,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 521,
  confidence: 0.6122,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_023' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.0162,
  latency: 28,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 1290,
  confidence: 0.7953,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_024' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.2042,
  latency: 236,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 8949,
  confidence: 0.0636,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_025' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_026' })
MERGE (a)-[r_025:MONITORS {
  strength: 0.718,
  latency: 57,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 6297,
  confidence: 0.5916,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_026' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.2299,
  latency: 145,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 940,
  confidence: 0.8937,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_027' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.5614,
  latency: 162,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 7387,
  confidence: 0.6354,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_028' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.8965,
  latency: 244,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 3635,
  confidence: 0.272,
  active: true
}]->(b);

MATCH (a:Multimodal { identifier: 'multimodal_05_metric_trackers_2_029' }),
      (b:Multimodal { identifier: 'multimodal_05_metric_trackers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.6033,
  latency: 168,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 2326,
  confidence: 0.4319,
  active: true
}]->(b);
