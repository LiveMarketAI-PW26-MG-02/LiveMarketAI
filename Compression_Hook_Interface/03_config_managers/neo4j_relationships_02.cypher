:param namespace => 'compression_02_02';
:param batchSize => 64;
:param threshold => 0.761;
:param maxDepth => 4;
:param timeoutSeconds => 70;
:param region => 'eu-west';
:param epoch => 16;
:param version => '1.6.2';

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_000' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.9749,
  latency: 219,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 379,
  confidence: 0.8627,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_001' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_002' })
MERGE (a)-[r_001:VALIDATES {
  strength: 0.0593,
  latency: 203,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 8393,
  confidence: 0.922,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_002' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.2898,
  latency: 148,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 3502,
  confidence: 0.5353,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_003' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.0096,
  latency: 110,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 1208,
  confidence: 0.6263,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_004' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.1453,
  latency: 126,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 8305,
  confidence: 0.5772,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_005' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.3706,
  latency: 224,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 9492,
  confidence: 0.8608,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_006' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.6255,
  latency: 233,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 5849,
  confidence: 0.9957,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_007' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.8885,
  latency: 127,
  established: datetime(),
  channel: 'channel_7',
  priority: 9,
  bandwidth: 7846,
  confidence: 0.7407,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_008' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.9544,
  latency: 78,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 2664,
  confidence: 0.0503,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_009' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.097,
  latency: 141,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 9408,
  confidence: 0.61,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_010' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.5654,
  latency: 86,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 2471,
  confidence: 0.0453,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_011' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.379,
  latency: 89,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 8520,
  confidence: 0.7679,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_012' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.4329,
  latency: 118,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 837,
  confidence: 0.3394,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_013' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.2102,
  latency: 156,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 484,
  confidence: 0.9525,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_014' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.6896,
  latency: 141,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 3698,
  confidence: 0.2959,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_015' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.5098,
  latency: 18,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 7337,
  confidence: 0.9861,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_016' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.3969,
  latency: 32,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 2005,
  confidence: 0.9663,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_017' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_018' })
MERGE (a)-[r_017:ROUTES_TO {
  strength: 0.1443,
  latency: 123,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 6478,
  confidence: 0.0536,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_018' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.7927,
  latency: 204,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 8171,
  confidence: 0.1829,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_019' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.2712,
  latency: 56,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 7361,
  confidence: 0.2255,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_020' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.1142,
  latency: 78,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 4099,
  confidence: 0.2759,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_021' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.1981,
  latency: 188,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 1311,
  confidence: 0.2122,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_022' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.6081,
  latency: 190,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 4972,
  confidence: 0.3619,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_023' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.4829,
  latency: 21,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 7953,
  confidence: 0.5816,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_024' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.9439,
  latency: 178,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 2733,
  confidence: 0.1081,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_025' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_026' })
MERGE (a)-[r_025:TRIGGERS {
  strength: 0.273,
  latency: 3,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 6947,
  confidence: 0.8834,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_026' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_027' })
MERGE (a)-[r_026:OBSERVES {
  strength: 0.2591,
  latency: 130,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 4399,
  confidence: 0.9841,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_027' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_028' })
MERGE (a)-[r_027:TRIGGERS {
  strength: 0.644,
  latency: 140,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 7848,
  confidence: 0.5651,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_028' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_029' })
MERGE (a)-[r_028:ROUTES_TO {
  strength: 0.5767,
  latency: 143,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 5279,
  confidence: 0.3198,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_03_config_managers_2_029' }),
      (b:Compression { identifier: 'compression_03_config_managers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.1259,
  latency: 95,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 8306,
  confidence: 0.266,
  active: true
}]->(b);
