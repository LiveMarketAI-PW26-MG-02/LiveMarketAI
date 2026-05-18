:param namespace => 'explainability_02_02';
:param batchSize => 256;
:param threshold => 0.66;
:param maxDepth => 10;
:param timeoutSeconds => 28;
:param region => 'eu-west';
:param epoch => 39;
:param version => '2.9.4';

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_000' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.1698,
  latency: 195,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 688,
  confidence: 0.026,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_001' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.3386,
  latency: 122,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 8936,
  confidence: 0.2994,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_002' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.2189,
  latency: 64,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 7681,
  confidence: 0.1764,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_003' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.8541,
  latency: 88,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 4542,
  confidence: 0.0412,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_004' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.1455,
  latency: 75,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 7846,
  confidence: 0.5969,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_005' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.6183,
  latency: 155,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 5274,
  confidence: 0.4969,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_006' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.0611,
  latency: 100,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 926,
  confidence: 0.9734,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_007' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.2758,
  latency: 126,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 8575,
  confidence: 0.5833,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_008' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.3336,
  latency: 98,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 1178,
  confidence: 0.907,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_009' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.7379,
  latency: 207,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 9734,
  confidence: 0.6434,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_010' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.0305,
  latency: 52,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 6663,
  confidence: 0.7976,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_011' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.9,
  latency: 180,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 6144,
  confidence: 0.5267,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_012' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.9729,
  latency: 203,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 954,
  confidence: 0.3622,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_013' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.6672,
  latency: 228,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 5917,
  confidence: 0.3057,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_014' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.5236,
  latency: 134,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 3734,
  confidence: 0.0082,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_015' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.6759,
  latency: 37,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 3525,
  confidence: 0.396,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_016' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.1003,
  latency: 18,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 2065,
  confidence: 0.292,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_017' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.1965,
  latency: 180,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 8238,
  confidence: 0.4489,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_018' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.1743,
  latency: 79,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 3495,
  confidence: 0.3206,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_019' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_020' })
MERGE (a)-[r_019:DEPENDS_ON {
  strength: 0.2781,
  latency: 63,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 2842,
  confidence: 0.164,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_020' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.3675,
  latency: 180,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 1176,
  confidence: 0.7624,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_021' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.3268,
  latency: 177,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 9414,
  confidence: 0.7722,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_022' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.7357,
  latency: 228,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 3194,
  confidence: 0.6822,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_023' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.1223,
  latency: 226,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 271,
  confidence: 0.4985,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_024' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.0341,
  latency: 213,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 641,
  confidence: 0.971,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_025' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.5054,
  latency: 220,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 6842,
  confidence: 0.089,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_026' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.0531,
  latency: 149,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4499,
  confidence: 0.0846,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_027' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.6688,
  latency: 100,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1744,
  confidence: 0.3546,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_028' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.9286,
  latency: 195,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 3395,
  confidence: 0.4287,
  active: true
}]->(b);

MATCH (a:Explainability { identifier: 'explainability_06_validation_layer_2_029' }),
      (b:Explainability { identifier: 'explainability_06_validation_layer_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.5269,
  latency: 250,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 6110,
  confidence: 0.2913,
  active: true
}]->(b);
