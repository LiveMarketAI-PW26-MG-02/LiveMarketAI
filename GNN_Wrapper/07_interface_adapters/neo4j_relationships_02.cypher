:param namespace => 'graphnetwork_02_02';
:param batchSize => 128;
:param threshold => 0.574;
:param maxDepth => 10;
:param timeoutSeconds => 87;
:param region => 'us-east';
:param epoch => 94;
:param version => '1.8.7';

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_000' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_001' })
MERGE (a)-[r_000:CALIBRATES {
  strength: 0.8284,
  latency: 144,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 3755,
  confidence: 0.0546,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_001' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.502,
  latency: 208,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 4990,
  confidence: 0.0025,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_002' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.0659,
  latency: 9,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 5875,
  confidence: 0.9798,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_003' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_004' })
MERGE (a)-[r_003:VALIDATES {
  strength: 0.0412,
  latency: 244,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 2939,
  confidence: 0.3423,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_004' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.7231,
  latency: 27,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 3891,
  confidence: 0.4513,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_005' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.5432,
  latency: 190,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 5938,
  confidence: 0.5594,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_006' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.0991,
  latency: 94,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 3665,
  confidence: 0.2904,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_007' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.4581,
  latency: 89,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 8223,
  confidence: 0.8408,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_008' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.6951,
  latency: 23,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 8615,
  confidence: 0.4478,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_009' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_010' })
MERGE (a)-[r_009:OBSERVES {
  strength: 0.0961,
  latency: 209,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 8127,
  confidence: 0.2335,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_010' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.744,
  latency: 31,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 3999,
  confidence: 0.0586,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_011' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.973,
  latency: 150,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 7741,
  confidence: 0.573,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_012' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.9431,
  latency: 166,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 7687,
  confidence: 0.8872,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_013' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_014' })
MERGE (a)-[r_013:CALIBRATES {
  strength: 0.7035,
  latency: 231,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 7413,
  confidence: 0.0578,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_014' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.5417,
  latency: 225,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 837,
  confidence: 0.3582,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_015' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.8223,
  latency: 182,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 9642,
  confidence: 0.587,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_016' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_017' })
MERGE (a)-[r_016:PRODUCES {
  strength: 0.8803,
  latency: 35,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 4702,
  confidence: 0.9876,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_017' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.0541,
  latency: 75,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 6665,
  confidence: 0.0177,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_018' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_019' })
MERGE (a)-[r_018:VALIDATES {
  strength: 0.273,
  latency: 202,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 8475,
  confidence: 0.1153,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_019' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.8603,
  latency: 42,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 418,
  confidence: 0.7148,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_020' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.7411,
  latency: 206,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 8654,
  confidence: 0.4458,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_021' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_022' })
MERGE (a)-[r_021:CALIBRATES {
  strength: 0.1935,
  latency: 182,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 2367,
  confidence: 0.3154,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_022' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_023' })
MERGE (a)-[r_022:CALIBRATES {
  strength: 0.4046,
  latency: 238,
  established: datetime(),
  channel: 'channel_6',
  priority: 10,
  bandwidth: 4369,
  confidence: 0.3557,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_023' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.0074,
  latency: 217,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 7990,
  confidence: 0.3653,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_024' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_025' })
MERGE (a)-[r_024:TRIGGERS {
  strength: 0.2498,
  latency: 173,
  established: datetime(),
  channel: 'channel_0',
  priority: 2,
  bandwidth: 7063,
  confidence: 0.3479,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_025' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.9416,
  latency: 229,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 8226,
  confidence: 0.1569,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_026' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.3673,
  latency: 146,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 3162,
  confidence: 0.6243,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_027' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.6738,
  latency: 196,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 8545,
  confidence: 0.1576,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_028' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_029' })
MERGE (a)-[r_028:OBSERVES {
  strength: 0.5208,
  latency: 229,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 436,
  confidence: 0.3847,
  active: true
}]->(b);

MATCH (a:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_029' }),
      (b:GraphNetwork { identifier: 'graphnetwork_07_interface_adapters_2_000' })
MERGE (a)-[r_029:ROUTES_TO {
  strength: 0.8846,
  latency: 31,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 1867,
  confidence: 0.1893,
  active: true
}]->(b);
