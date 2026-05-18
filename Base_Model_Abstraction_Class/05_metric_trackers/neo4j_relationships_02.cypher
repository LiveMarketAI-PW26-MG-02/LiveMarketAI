:param namespace => 'basemodel_02_02';
:param batchSize => 256;
:param threshold => 0.729;
:param maxDepth => 3;
:param timeoutSeconds => 21;
:param region => 'eu-west';
:param epoch => 19;
:param version => '5.2.2';

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_000' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_001' })
MERGE (a)-[r_000:TRIGGERS {
  strength: 0.7702,
  latency: 53,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 3698,
  confidence: 0.686,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_001' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.9973,
  latency: 203,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 718,
  confidence: 0.4973,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_002' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.9081,
  latency: 193,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 1126,
  confidence: 0.199,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_003' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.8457,
  latency: 202,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1613,
  confidence: 0.6509,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_004' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_005' })
MERGE (a)-[r_004:TRIGGERS {
  strength: 0.5828,
  latency: 206,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 8230,
  confidence: 0.1349,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_005' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.9048,
  latency: 191,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 9772,
  confidence: 0.1647,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_006' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.8252,
  latency: 201,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 4998,
  confidence: 0.748,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_007' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.068,
  latency: 201,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 3902,
  confidence: 0.2401,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_008' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.5616,
  latency: 225,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 9520,
  confidence: 0.9081,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_009' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.392,
  latency: 201,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 5713,
  confidence: 0.8258,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_010' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.9481,
  latency: 59,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 9846,
  confidence: 0.9041,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_011' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.7929,
  latency: 2,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 8112,
  confidence: 0.6038,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_012' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.8782,
  latency: 122,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 6830,
  confidence: 0.6048,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_013' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.1458,
  latency: 140,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 1461,
  confidence: 0.3537,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_014' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.6193,
  latency: 75,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 1541,
  confidence: 0.9925,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_015' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.7012,
  latency: 114,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 8917,
  confidence: 0.8071,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_016' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_017' })
MERGE (a)-[r_016:PRODUCES {
  strength: 0.2163,
  latency: 161,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 6254,
  confidence: 0.8229,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_017' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.3897,
  latency: 86,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 6037,
  confidence: 0.1674,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_018' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.8899,
  latency: 157,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 5155,
  confidence: 0.4997,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_019' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.8567,
  latency: 248,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 6505,
  confidence: 0.5272,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_020' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_021' })
MERGE (a)-[r_020:DEPENDS_ON {
  strength: 0.8535,
  latency: 27,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7547,
  confidence: 0.5653,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_021' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.7366,
  latency: 174,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 9155,
  confidence: 0.7346,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_022' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.135,
  latency: 193,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 6916,
  confidence: 0.0759,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_023' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.4441,
  latency: 246,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 6028,
  confidence: 0.3053,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_024' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.9378,
  latency: 208,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 8261,
  confidence: 0.4933,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_025' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.057,
  latency: 214,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 9232,
  confidence: 0.3772,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_026' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.7511,
  latency: 229,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 7618,
  confidence: 0.0351,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_027' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.4825,
  latency: 2,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 2467,
  confidence: 0.1877,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_028' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.9977,
  latency: 45,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 4701,
  confidence: 0.6273,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_05_metric_trackers_2_029' }),
      (b:BaseModel { identifier: 'basemodel_05_metric_trackers_2_000' })
MERGE (a)-[r_029:VALIDATES {
  strength: 0.2912,
  latency: 140,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 6992,
  confidence: 0.5482,
  active: true
}]->(b);
