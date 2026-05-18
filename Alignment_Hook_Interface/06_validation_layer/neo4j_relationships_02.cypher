:param namespace => 'alignment_02_02';
:param batchSize => 128;
:param threshold => 0.174;
:param maxDepth => 10;
:param timeoutSeconds => 18;
:param region => 'us-east';
:param epoch => 34;
:param version => '3.1.4';

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_000' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.4958,
  latency: 170,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 3403,
  confidence: 0.7427,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_001' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_002' })
MERGE (a)-[r_001:VALIDATES {
  strength: 0.675,
  latency: 193,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 7204,
  confidence: 0.6659,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_002' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.273,
  latency: 133,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 2936,
  confidence: 0.4509,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_003' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.146,
  latency: 80,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 8259,
  confidence: 0.019,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_004' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.7563,
  latency: 208,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 2380,
  confidence: 0.026,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_005' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.3939,
  latency: 35,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 766,
  confidence: 0.6055,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_006' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.1681,
  latency: 21,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 6123,
  confidence: 0.9267,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_007' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_008' })
MERGE (a)-[r_007:PRODUCES {
  strength: 0.5968,
  latency: 106,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 7637,
  confidence: 0.5555,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_008' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.9377,
  latency: 21,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 3924,
  confidence: 0.2132,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_009' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.6609,
  latency: 92,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 8894,
  confidence: 0.1264,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_010' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.0703,
  latency: 232,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 7548,
  confidence: 0.2487,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_011' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.5942,
  latency: 41,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 4031,
  confidence: 0.021,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_012' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.5604,
  latency: 146,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 8184,
  confidence: 0.6232,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_013' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.8577,
  latency: 78,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 9976,
  confidence: 0.3288,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_014' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.1734,
  latency: 229,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 234,
  confidence: 0.0988,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_015' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.887,
  latency: 69,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6402,
  confidence: 0.2378,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_016' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.9618,
  latency: 222,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 7125,
  confidence: 0.1169,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_017' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.9129,
  latency: 71,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 3350,
  confidence: 0.0275,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_018' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.4848,
  latency: 114,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 6649,
  confidence: 0.5313,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_019' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_020' })
MERGE (a)-[r_019:CALIBRATES {
  strength: 0.6172,
  latency: 129,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 1231,
  confidence: 0.1089,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_020' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.0132,
  latency: 12,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 6268,
  confidence: 0.9163,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_021' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.9043,
  latency: 24,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 5251,
  confidence: 0.1086,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_022' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.3333,
  latency: 85,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 7775,
  confidence: 0.4793,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_023' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.2138,
  latency: 208,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 6027,
  confidence: 0.4739,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_024' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.6264,
  latency: 246,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 3721,
  confidence: 0.3687,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_025' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.0355,
  latency: 31,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 4602,
  confidence: 0.4075,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_026' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.5755,
  latency: 30,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 3936,
  confidence: 0.6371,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_027' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.2182,
  latency: 13,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 7415,
  confidence: 0.9379,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_028' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.3531,
  latency: 188,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 7236,
  confidence: 0.8679,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_06_validation_layer_2_029' }),
      (b:Alignment { identifier: 'alignment_06_validation_layer_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.6239,
  latency: 124,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 3764,
  confidence: 0.3956,
  active: true
}]->(b);
