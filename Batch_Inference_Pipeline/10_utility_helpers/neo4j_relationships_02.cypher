:param namespace => 'batchinference_02_02';
:param batchSize => 256;
:param threshold => 0.714;
:param maxDepth => 9;
:param timeoutSeconds => 13;
:param region => 'ap-south';
:param epoch => 26;
:param version => '4.1.2';

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_000' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.3261,
  latency: 77,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 5592,
  confidence: 0.8508,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_001' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.6982,
  latency: 79,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 6133,
  confidence: 0.4697,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_002' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_003' })
MERGE (a)-[r_002:MONITORS {
  strength: 0.0057,
  latency: 138,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 3829,
  confidence: 0.5072,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_003' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_004' })
MERGE (a)-[r_003:PRODUCES {
  strength: 0.1819,
  latency: 81,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 1907,
  confidence: 0.73,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_004' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.3648,
  latency: 47,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 9918,
  confidence: 0.6728,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_005' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.3647,
  latency: 198,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 8266,
  confidence: 0.5395,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_006' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.3371,
  latency: 213,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 8334,
  confidence: 0.6695,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_007' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_008' })
MERGE (a)-[r_007:ROUTES_TO {
  strength: 0.0166,
  latency: 100,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 7098,
  confidence: 0.0803,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_008' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_009' })
MERGE (a)-[r_008:VALIDATES {
  strength: 0.3852,
  latency: 53,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 4088,
  confidence: 0.3297,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_009' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.399,
  latency: 229,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 3088,
  confidence: 0.2762,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_010' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.1995,
  latency: 184,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 7919,
  confidence: 0.235,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_011' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.4249,
  latency: 37,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 1601,
  confidence: 0.733,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_012' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.9756,
  latency: 54,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 9819,
  confidence: 0.7131,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_013' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_014' })
MERGE (a)-[r_013:DEPENDS_ON {
  strength: 0.3198,
  latency: 86,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 3843,
  confidence: 0.1697,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_014' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.2098,
  latency: 250,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 1396,
  confidence: 0.8171,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_015' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.6083,
  latency: 241,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 1778,
  confidence: 0.3133,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_016' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.3212,
  latency: 234,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 7392,
  confidence: 0.0087,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_017' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.1185,
  latency: 36,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 7871,
  confidence: 0.1534,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_018' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.4263,
  latency: 86,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 8030,
  confidence: 0.8463,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_019' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.8111,
  latency: 154,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 1807,
  confidence: 0.5307,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_020' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.9097,
  latency: 89,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 5815,
  confidence: 0.9688,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_021' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.5765,
  latency: 194,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 7305,
  confidence: 0.5864,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_022' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.2935,
  latency: 117,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 8551,
  confidence: 0.5789,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_023' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.8873,
  latency: 104,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 3463,
  confidence: 0.0886,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_024' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.5822,
  latency: 134,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 889,
  confidence: 0.6053,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_025' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_026' })
MERGE (a)-[r_025:PRODUCES {
  strength: 0.406,
  latency: 31,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 7219,
  confidence: 0.1781,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_026' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.0913,
  latency: 87,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 881,
  confidence: 0.1089,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_027' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.9455,
  latency: 193,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 2532,
  confidence: 0.3674,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_028' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.4769,
  latency: 71,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 9599,
  confidence: 0.5178,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_10_utility_helpers_2_029' }),
      (b:BatchInference { identifier: 'batchinference_10_utility_helpers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.1205,
  latency: 153,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 4687,
  confidence: 0.2192,
  active: true
}]->(b);
