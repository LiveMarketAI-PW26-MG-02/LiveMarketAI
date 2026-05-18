:param namespace => 'batchinference_02_02';
:param batchSize => 512;
:param threshold => 0.459;
:param maxDepth => 6;
:param timeoutSeconds => 44;
:param region => 'ap-south';
:param epoch => 36;
:param version => '1.8.8';

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_000' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.2428,
  latency: 47,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 8025,
  confidence: 0.5726,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_001' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.5214,
  latency: 211,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 7921,
  confidence: 0.6229,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_002' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_003' })
MERGE (a)-[r_002:PRODUCES {
  strength: 0.3715,
  latency: 227,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 6907,
  confidence: 0.6208,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_003' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.0678,
  latency: 228,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 7067,
  confidence: 0.747,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_004' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.7655,
  latency: 182,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 7440,
  confidence: 0.6686,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_005' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_006' })
MERGE (a)-[r_005:TRIGGERS {
  strength: 0.5168,
  latency: 250,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 599,
  confidence: 0.1859,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_006' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.479,
  latency: 71,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 8248,
  confidence: 0.708,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_007' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_008' })
MERGE (a)-[r_007:TRIGGERS {
  strength: 0.8079,
  latency: 47,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 1189,
  confidence: 0.266,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_008' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.4479,
  latency: 75,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 4712,
  confidence: 0.3838,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_009' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.0742,
  latency: 235,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 6133,
  confidence: 0.3656,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_010' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_011' })
MERGE (a)-[r_010:VALIDATES {
  strength: 0.8028,
  latency: 73,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 2845,
  confidence: 0.6497,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_011' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_012' })
MERGE (a)-[r_011:VALIDATES {
  strength: 0.1938,
  latency: 161,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 543,
  confidence: 0.2818,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_012' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.5788,
  latency: 85,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 2676,
  confidence: 0.2546,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_013' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_014' })
MERGE (a)-[r_013:PRODUCES {
  strength: 0.5552,
  latency: 91,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 1974,
  confidence: 0.3954,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_014' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.2692,
  latency: 30,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 605,
  confidence: 0.1294,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_015' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.2216,
  latency: 111,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 3204,
  confidence: 0.5985,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_016' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_017' })
MERGE (a)-[r_016:MONITORS {
  strength: 0.0382,
  latency: 131,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 5862,
  confidence: 0.6266,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_017' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_018' })
MERGE (a)-[r_017:PRODUCES {
  strength: 0.608,
  latency: 146,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 4991,
  confidence: 0.9028,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_018' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_019' })
MERGE (a)-[r_018:DEPENDS_ON {
  strength: 0.6586,
  latency: 212,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 980,
  confidence: 0.4383,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_019' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_020' })
MERGE (a)-[r_019:PRODUCES {
  strength: 0.3403,
  latency: 116,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 3870,
  confidence: 0.5257,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_020' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.4816,
  latency: 7,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 6076,
  confidence: 0.9244,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_021' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.2065,
  latency: 104,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 4180,
  confidence: 0.2199,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_022' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.0384,
  latency: 22,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 6054,
  confidence: 0.5939,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_023' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.5736,
  latency: 124,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 5216,
  confidence: 0.7734,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_024' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.1238,
  latency: 94,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 2455,
  confidence: 0.9309,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_025' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_026' })
MERGE (a)-[r_025:CALIBRATES {
  strength: 0.4478,
  latency: 232,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 6940,
  confidence: 0.7277,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_026' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.8644,
  latency: 190,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 3605,
  confidence: 0.7529,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_027' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.3672,
  latency: 123,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 7292,
  confidence: 0.0585,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_028' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.7173,
  latency: 196,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 8228,
  confidence: 0.6567,
  active: true
}]->(b);

MATCH (a:BatchInference { identifier: 'batchinference_03_config_managers_2_029' }),
      (b:BatchInference { identifier: 'batchinference_03_config_managers_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.1225,
  latency: 14,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 7958,
  confidence: 0.4885,
  active: true
}]->(b);
