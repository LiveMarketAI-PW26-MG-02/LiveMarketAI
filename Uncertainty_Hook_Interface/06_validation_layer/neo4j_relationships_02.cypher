:param namespace => 'uncertainty_02_02';
:param batchSize => 256;
:param threshold => 0.629;
:param maxDepth => 12;
:param timeoutSeconds => 50;
:param region => 'eu-west';
:param epoch => 56;
:param version => '4.6.8';

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_000' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.5826,
  latency: 42,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 4270,
  confidence: 0.8955,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_001' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_002' })
MERGE (a)-[r_001:VALIDATES {
  strength: 0.8029,
  latency: 183,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 6773,
  confidence: 0.7228,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_002' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_003' })
MERGE (a)-[r_002:OBSERVES {
  strength: 0.2576,
  latency: 126,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 549,
  confidence: 0.6733,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_003' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.5439,
  latency: 150,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 799,
  confidence: 0.8058,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_004' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.1065,
  latency: 246,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 638,
  confidence: 0.8814,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_005' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.1974,
  latency: 81,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 7996,
  confidence: 0.6841,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_006' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_007' })
MERGE (a)-[r_006:TRIGGERS {
  strength: 0.4305,
  latency: 165,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 3686,
  confidence: 0.8612,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_007' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.3285,
  latency: 240,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 571,
  confidence: 0.0185,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_008' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_009' })
MERGE (a)-[r_008:TRIGGERS {
  strength: 0.8162,
  latency: 178,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 9301,
  confidence: 0.1499,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_009' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.1328,
  latency: 212,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 133,
  confidence: 0.1768,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_010' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.7049,
  latency: 246,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 6241,
  confidence: 0.4643,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_011' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_012' })
MERGE (a)-[r_011:TRIGGERS {
  strength: 0.5342,
  latency: 196,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 7701,
  confidence: 0.9308,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_012' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.0923,
  latency: 200,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 310,
  confidence: 0.8601,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_013' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.6403,
  latency: 241,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 8212,
  confidence: 0.3222,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_014' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.5032,
  latency: 248,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 8981,
  confidence: 0.7869,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_015' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_016' })
MERGE (a)-[r_015:OBSERVES {
  strength: 0.231,
  latency: 239,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 2868,
  confidence: 0.102,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_016' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.7048,
  latency: 171,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 3870,
  confidence: 0.7491,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_017' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.8547,
  latency: 231,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 5492,
  confidence: 0.0136,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_018' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.2119,
  latency: 181,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 2115,
  confidence: 0.4655,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_019' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_020' })
MERGE (a)-[r_019:CALIBRATES {
  strength: 0.3011,
  latency: 93,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 2677,
  confidence: 0.0534,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_020' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.9887,
  latency: 113,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 1202,
  confidence: 0.6986,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_021' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.9021,
  latency: 21,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 641,
  confidence: 0.2311,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_022' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.8262,
  latency: 138,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 2325,
  confidence: 0.4411,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_023' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_024' })
MERGE (a)-[r_023:CALIBRATES {
  strength: 0.6194,
  latency: 111,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 1317,
  confidence: 0.13,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_024' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.5705,
  latency: 69,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 9107,
  confidence: 0.298,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_025' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.6939,
  latency: 229,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 9927,
  confidence: 0.2526,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_026' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_027' })
MERGE (a)-[r_026:MONITORS {
  strength: 0.737,
  latency: 61,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 2362,
  confidence: 0.446,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_027' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_028' })
MERGE (a)-[r_027:VALIDATES {
  strength: 0.8812,
  latency: 192,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 8906,
  confidence: 0.5211,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_028' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.8234,
  latency: 185,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 5805,
  confidence: 0.5325,
  active: true
}]->(b);

MATCH (a:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_029' }),
      (b:Uncertainty { identifier: 'uncertainty_06_validation_layer_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.2678,
  latency: 146,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 9036,
  confidence: 0.3117,
  active: true
}]->(b);
