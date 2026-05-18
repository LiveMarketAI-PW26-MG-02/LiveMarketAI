:param namespace => 'inferencecontext_02_02';
:param batchSize => 512;
:param threshold => 0.493;
:param maxDepth => 4;
:param timeoutSeconds => 26;
:param region => 'eu-west';
:param epoch => 19;
:param version => '3.8.2';

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_000' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.8196,
  latency: 9,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 9688,
  confidence: 0.5808,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_001' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.9993,
  latency: 20,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 9631,
  confidence: 0.2335,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_002' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.9386,
  latency: 218,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 8391,
  confidence: 0.2752,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_003' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.325,
  latency: 233,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 7251,
  confidence: 0.2854,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_004' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.8694,
  latency: 73,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 1166,
  confidence: 0.1639,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_005' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.414,
  latency: 229,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 3199,
  confidence: 0.806,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_006' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.9588,
  latency: 223,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 6458,
  confidence: 0.4443,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_007' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.4021,
  latency: 241,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 2385,
  confidence: 0.5621,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_008' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.9029,
  latency: 133,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 7940,
  confidence: 0.8389,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_009' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_010' })
MERGE (a)-[r_009:PRODUCES {
  strength: 0.7134,
  latency: 42,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 1402,
  confidence: 0.3404,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_010' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.5337,
  latency: 84,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 7732,
  confidence: 0.4964,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_011' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.3514,
  latency: 202,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 7040,
  confidence: 0.5509,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_012' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_013' })
MERGE (a)-[r_012:VALIDATES {
  strength: 0.1783,
  latency: 145,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 2905,
  confidence: 0.3166,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_013' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.8952,
  latency: 17,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 7532,
  confidence: 0.4884,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_014' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.6221,
  latency: 78,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 624,
  confidence: 0.0463,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_015' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.4453,
  latency: 194,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 4886,
  confidence: 0.5693,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_016' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.8508,
  latency: 191,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 4553,
  confidence: 0.3245,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_017' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.7637,
  latency: 64,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 9166,
  confidence: 0.3538,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_018' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.0761,
  latency: 199,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 5366,
  confidence: 0.4595,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_019' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_020' })
MERGE (a)-[r_019:TRIGGERS {
  strength: 0.7453,
  latency: 239,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 7325,
  confidence: 0.2165,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_020' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.9502,
  latency: 219,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 4376,
  confidence: 0.7153,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_021' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_022' })
MERGE (a)-[r_021:MONITORS {
  strength: 0.3066,
  latency: 248,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 386,
  confidence: 0.8553,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_022' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.2434,
  latency: 199,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 6132,
  confidence: 0.4031,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_023' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.7014,
  latency: 214,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 9277,
  confidence: 0.1086,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_024' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.6596,
  latency: 194,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 1798,
  confidence: 0.5969,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_025' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.608,
  latency: 165,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 5781,
  confidence: 0.4517,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_026' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.7796,
  latency: 40,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 3620,
  confidence: 0.9476,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_027' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.0066,
  latency: 183,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 5320,
  confidence: 0.016,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_028' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.2004,
  latency: 153,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 9884,
  confidence: 0.1298,
  active: true
}]->(b);

MATCH (a:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_029' }),
      (b:InferenceContext { identifier: 'inferencecontext_06_validation_layer_2_000' })
MERGE (a)-[r_029:OBSERVES {
  strength: 0.6698,
  latency: 35,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3691,
  confidence: 0.733,
  active: true
}]->(b);
