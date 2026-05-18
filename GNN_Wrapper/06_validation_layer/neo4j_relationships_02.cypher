:param namespace => 'graphnetwork_02_02';
:param batchSize => 512;
:param threshold => 0.832;
:param maxDepth => 5;
:param timeoutSeconds => 50;
:param region => 'eu-west';
:param epoch => 98;
:param version => '3.7.2';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.6722,
  latency: 199,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 445,
  confidence: 0.898,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.1005,
  latency: 10,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6835,
  confidence: 0.1445,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.7065,
  latency: 123,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9509,
  confidence: 0.4218,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.3851,
  latency: 2,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 3422,
  confidence: 0.0562,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.6574,
  latency: 10,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 8857,
  confidence: 0.4394,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.6797,
  latency: 129,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 6477,
  confidence: 0.3185,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_007' })
MERGE (a)-[r_006:PRODUCES {
  strength: 0.2098,
  latency: 89,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 8880,
  confidence: 0.787,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.5643,
  latency: 154,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 2331,
  confidence: 0.6933,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.8714,
  latency: 7,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 7242,
  confidence: 0.4853,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.3215,
  latency: 25,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 4226,
  confidence: 0.0721,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.5617,
  latency: 123,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 6590,
  confidence: 0.9512,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.5195,
  latency: 23,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 6386,
  confidence: 0.0871,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.0587,
  latency: 22,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 4918,
  confidence: 0.6954,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.2647,
  latency: 180,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 3798,
  confidence: 0.991,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_015' })
MERGE (a)-[r_014:OBSERVES {
  strength: 0.1462,
  latency: 131,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 8622,
  confidence: 0.1552,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.1481,
  latency: 69,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 9275,
  confidence: 0.0257,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.5983,
  latency: 67,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 4805,
  confidence: 0.3249,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.4489,
  latency: 91,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 2887,
  confidence: 0.6363,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_019' })
MERGE (a)-[r_018:CALIBRATES {
  strength: 0.585,
  latency: 14,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 9965,
  confidence: 0.9676,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.0874,
  latency: 29,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 7692,
  confidence: 0.7215,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.5035,
  latency: 13,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 4957,
  confidence: 0.4958,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.0408,
  latency: 193,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 125,
  confidence: 0.9217,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.0592,
  latency: 172,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 4824,
  confidence: 0.2926,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.7292,
  latency: 73,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 8468,
  confidence: 0.2576,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.617,
  latency: 56,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 9948,
  confidence: 0.7334,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.2553,
  latency: 218,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 1593,
  confidence: 0.7198,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.6158,
  latency: 215,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6636,
  confidence: 0.0482,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.5171,
  latency: 101,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 7426,
  confidence: 0.8903,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.3726,
  latency: 156,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 8927,
  confidence: 0.3407,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_06_validation_layer_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.2182,
  latency: 58,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 773,
  confidence: 0.9625,
  active: true
}]->(b);
