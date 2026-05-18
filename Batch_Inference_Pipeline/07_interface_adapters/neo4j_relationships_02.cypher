:param namespace => 'batchinference_02_02';
:param batchSize => 128;
:param threshold => 0.834;
:param maxDepth => 4;
:param timeoutSeconds => 71;
:param region => 'eu-west';
:param epoch => 96;
:param version => '2.7.1';

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_000' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_001' })
MERGE (a)-[r_000:PRODUCES {
  strength: 0.558,
  latency: 85,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 6669,
  confidence: 0.2278,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_001' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.4004,
  latency: 78,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 8762,
  confidence: 0.6779,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_002' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_003' })
MERGE (a)-[r_002:CALIBRATES {
  strength: 0.1471,
  latency: 5,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 931,
  confidence: 0.4163,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_003' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.0334,
  latency: 158,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 887,
  confidence: 0.4099,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_004' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.5165,
  latency: 132,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 2305,
  confidence: 0.7299,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_005' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.9511,
  latency: 68,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 897,
  confidence: 0.8532,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_006' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.079,
  latency: 104,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 779,
  confidence: 0.9934,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_007' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.3395,
  latency: 11,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 2508,
  confidence: 0.6479,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_008' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.5357,
  latency: 88,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 9748,
  confidence: 0.1019,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_009' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.6082,
  latency: 122,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 3519,
  confidence: 0.926,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_010' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.0189,
  latency: 199,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 9188,
  confidence: 0.5923,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_011' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.3504,
  latency: 124,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 5438,
  confidence: 0.0371,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_012' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.117,
  latency: 59,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 256,
  confidence: 0.4604,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_013' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.4414,
  latency: 54,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 5884,
  confidence: 0.5579,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_014' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.2094,
  latency: 35,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 895,
  confidence: 0.7522,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_015' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_016' })
MERGE (a)-[r_015:DEPENDS_ON {
  strength: 0.6603,
  latency: 176,
  established: datetime(),
  channel: 'channel_7',
  priority: 8,
  bandwidth: 9841,
  confidence: 0.3106,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_016' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.3271,
  latency: 43,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 286,
  confidence: 0.7023,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_017' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.7097,
  latency: 195,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 2037,
  confidence: 0.2755,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_018' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.6142,
  latency: 37,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 6636,
  confidence: 0.1495,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_019' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.1921,
  latency: 240,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 7155,
  confidence: 0.7692,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_020' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_021' })
MERGE (a)-[r_020:PRODUCES {
  strength: 0.6144,
  latency: 213,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 1291,
  confidence: 0.9512,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_021' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.3922,
  latency: 83,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 6466,
  confidence: 0.1165,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_022' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.0513,
  latency: 17,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 8503,
  confidence: 0.0179,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_023' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.216,
  latency: 159,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 2116,
  confidence: 0.7854,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_024' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_025' })
MERGE (a)-[r_024:PRODUCES {
  strength: 0.7072,
  latency: 236,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 5354,
  confidence: 0.1921,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_025' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.9564,
  latency: 229,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 1116,
  confidence: 0.1044,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_026' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.2338,
  latency: 134,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 9316,
  confidence: 0.2898,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_027' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.2158,
  latency: 114,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 5107,
  confidence: 0.1468,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_028' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_029' })
MERGE (a)-[r_028:MONITORS {
  strength: 0.7816,
  latency: 83,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 5869,
  confidence: 0.2212,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_07_interface_adapters_2_029' }),
      (b:BatchInference { identifier: 'batchinference_07_interface_adapters_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.9002,
  latency: 239,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 9825,
  confidence: 0.4248,
  active: true
}]->(b);
