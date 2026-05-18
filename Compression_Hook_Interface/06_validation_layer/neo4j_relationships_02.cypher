:param namespace => 'compression_02_02';
:param batchSize => 256;
:param threshold => 0.351;
:param maxDepth => 5;
:param timeoutSeconds => 32;
:param region => 'us-east';
:param epoch => 77;
:param version => '5.3.2';

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_000' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.3894,
  latency: 18,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 3455,
  confidence: 0.4486,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_001' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.2318,
  latency: 177,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 8826,
  confidence: 0.6345,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_002' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_003' })
MERGE (a)-[r_002:OBSERVES {
  strength: 0.3305,
  latency: 119,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6290,
  confidence: 0.8415,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_003' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.9935,
  latency: 236,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 8924,
  confidence: 0.2811,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_004' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.0958,
  latency: 208,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 6825,
  confidence: 0.282,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_005' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.305,
  latency: 140,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2461,
  confidence: 0.7668,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_006' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.6861,
  latency: 115,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 5196,
  confidence: 0.4034,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_007' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.3502,
  latency: 33,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 5566,
  confidence: 0.8469,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_008' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.4151,
  latency: 237,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 9559,
  confidence: 0.449,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_009' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_010' })
MERGE (a)-[r_009:VALIDATES {
  strength: 0.3918,
  latency: 54,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 1556,
  confidence: 0.6734,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_010' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.523,
  latency: 52,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 5313,
  confidence: 0.7428,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_011' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_012' })
MERGE (a)-[r_011:DEPENDS_ON {
  strength: 0.6209,
  latency: 217,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 8922,
  confidence: 0.8115,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_012' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.676,
  latency: 157,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 1467,
  confidence: 0.6799,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_013' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.3973,
  latency: 4,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 5648,
  confidence: 0.459,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_014' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.5113,
  latency: 111,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 4854,
  confidence: 0.2767,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_015' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.7574,
  latency: 162,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 569,
  confidence: 0.2741,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_016' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.5811,
  latency: 186,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 7472,
  confidence: 0.9949,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_017' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.085,
  latency: 4,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 1409,
  confidence: 0.8068,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_018' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.2804,
  latency: 194,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 7139,
  confidence: 0.2134,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_019' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.742,
  latency: 66,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 9807,
  confidence: 0.4126,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_020' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.0622,
  latency: 227,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 4354,
  confidence: 0.6587,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_021' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_022' })
MERGE (a)-[r_021:ROUTES_TO {
  strength: 0.8517,
  latency: 76,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 3399,
  confidence: 0.3895,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_022' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.2032,
  latency: 92,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 3024,
  confidence: 0.0931,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_023' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.796,
  latency: 222,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 409,
  confidence: 0.6862,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_024' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.7038,
  latency: 153,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 722,
  confidence: 0.9644,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_025' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.2981,
  latency: 236,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 5172,
  confidence: 0.5168,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_026' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_027' })
MERGE (a)-[r_026:TRIGGERS {
  strength: 0.1353,
  latency: 153,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 1317,
  confidence: 0.4429,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_027' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.6132,
  latency: 224,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 4347,
  confidence: 0.8028,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_028' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.4117,
  latency: 186,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 2179,
  confidence: 0.4513,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_06_validation_layer_2_029' }),
      (b:Compression { identifier: 'compression_06_validation_layer_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.441,
  latency: 26,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 6088,
  confidence: 0.4032,
  active: true
}]->(b);
