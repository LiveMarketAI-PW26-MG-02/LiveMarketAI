:param namespace => 'graphnetwork_02_02';
:param batchSize => 64;
:param threshold => 0.46;
:param maxDepth => 4;
:param timeoutSeconds => 43;
:param region => 'us-east';
:param epoch => 54;
:param version => '3.0.7';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.1492,
  latency: 236,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 6781,
  confidence: 0.4179,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.3242,
  latency: 250,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 7244,
  confidence: 0.561,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.683,
  latency: 210,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 5724,
  confidence: 0.9195,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:PRODUCES {
  strength: 0.245,
  latency: 8,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 5550,
  confidence: 0.2605,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.1412,
  latency: 103,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 7763,
  confidence: 0.9522,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.843,
  latency: 190,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 9680,
  confidence: 0.8045,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.4689,
  latency: 41,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 4249,
  confidence: 0.2101,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.0688,
  latency: 136,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 4641,
  confidence: 0.7332,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.851,
  latency: 60,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 5447,
  confidence: 0.4921,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.2033,
  latency: 11,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 3984,
  confidence: 0.4875,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.9435,
  latency: 130,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 3166,
  confidence: 0.3851,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.4774,
  latency: 132,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 8379,
  confidence: 0.9658,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.114,
  latency: 201,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 431,
  confidence: 0.6412,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.3444,
  latency: 134,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 9663,
  confidence: 0.0935,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.3116,
  latency: 98,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 9519,
  confidence: 0.8163,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:VALIDATES {
  strength: 0.763,
  latency: 241,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 5429,
  confidence: 0.2378,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.9945,
  latency: 111,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 2223,
  confidence: 0.7027,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.85,
  latency: 183,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 2140,
  confidence: 0.9059,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.872,
  latency: 66,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 7962,
  confidence: 0.9313,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.0975,
  latency: 62,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 2839,
  confidence: 0.7765,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.4681,
  latency: 215,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 1420,
  confidence: 0.5277,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.0583,
  latency: 173,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 8039,
  confidence: 0.4108,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.4308,
  latency: 141,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 7128,
  confidence: 0.1897,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.8183,
  latency: 237,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 9887,
  confidence: 0.1463,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.692,
  latency: 120,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 5755,
  confidence: 0.6667,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.5306,
  latency: 177,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 130,
  confidence: 0.9961,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.4978,
  latency: 224,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 9969,
  confidence: 0.8378,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.4951,
  latency: 52,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 3607,
  confidence: 0.2578,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.0027,
  latency: 163,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 6426,
  confidence: 0.2153,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.198,
  latency: 247,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 8101,
  confidence: 0.2744,
  active: true
}]->(b);
