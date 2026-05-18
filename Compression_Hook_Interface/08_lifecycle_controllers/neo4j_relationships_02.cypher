:param namespace => 'compression_02_02';
:param batchSize => 512;
:param threshold => 0.816;
:param maxDepth => 11;
:param timeoutSeconds => 65;
:param region => 'eu-west';
:param epoch => 19;
:param version => '4.6.6';

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_000' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.4284,
  latency: 246,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 9694,
  confidence: 0.6248,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_001' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:OBSERVES {
  strength: 0.5053,
  latency: 139,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 9089,
  confidence: 0.1629,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_002' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:OBSERVES {
  strength: 0.4096,
  latency: 17,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 7436,
  confidence: 0.3429,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_003' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.6861,
  latency: 71,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 2094,
  confidence: 0.1524,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_004' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:CALIBRATES {
  strength: 0.043,
  latency: 201,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 2452,
  confidence: 0.7384,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_005' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.4245,
  latency: 146,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 6410,
  confidence: 0.4069,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_006' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:CALIBRATES {
  strength: 0.5437,
  latency: 215,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 3505,
  confidence: 0.4427,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_007' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.1799,
  latency: 76,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 2034,
  confidence: 0.8401,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_008' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:OBSERVES {
  strength: 0.8072,
  latency: 217,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 9111,
  confidence: 0.8406,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_009' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.1391,
  latency: 121,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 7807,
  confidence: 0.629,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_010' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:TRIGGERS {
  strength: 0.2115,
  latency: 153,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 3899,
  confidence: 0.8591,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_011' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.3232,
  latency: 19,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 3960,
  confidence: 0.4429,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_012' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:OBSERVES {
  strength: 0.0036,
  latency: 25,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 4842,
  confidence: 0.9883,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_013' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.522,
  latency: 138,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 1271,
  confidence: 0.1317,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_014' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:TRIGGERS {
  strength: 0.2752,
  latency: 240,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 3815,
  confidence: 0.4608,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_015' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.3702,
  latency: 24,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 1738,
  confidence: 0.9319,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_016' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:VALIDATES {
  strength: 0.4546,
  latency: 165,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 5928,
  confidence: 0.3576,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_017' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.0803,
  latency: 232,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 3905,
  confidence: 0.5909,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_018' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.1258,
  latency: 164,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 4690,
  confidence: 0.2096,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_019' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.5491,
  latency: 250,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 869,
  confidence: 0.5976,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_020' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:MONITORS {
  strength: 0.4435,
  latency: 190,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 713,
  confidence: 0.8258,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_021' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.6067,
  latency: 199,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 4807,
  confidence: 0.7713,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_022' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.6589,
  latency: 105,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 4222,
  confidence: 0.4195,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_023' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.3896,
  latency: 194,
  established: datetime(),
  channel: 'channel_7',
  priority: 5,
  bandwidth: 4424,
  confidence: 0.7518,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_024' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.6591,
  latency: 110,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 6037,
  confidence: 0.3448,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_025' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.1369,
  latency: 172,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 389,
  confidence: 0.595,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_026' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.9315,
  latency: 194,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 1676,
  confidence: 0.0008,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_027' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:CALIBRATES {
  strength: 0.7268,
  latency: 186,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 3941,
  confidence: 0.1785,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_028' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.5815,
  latency: 84,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 8367,
  confidence: 0.6146,
  active: true
}]->(b);

MATCH (a:Compression { identifier: 'compression_08_lifecycle_controllers_2_029' }),
      (b:Compression { identifier: 'compression_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.4913,
  latency: 207,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 2382,
  confidence: 0.9027,
  active: true
}]->(b);
