:param namespace => 'checkpointloader_02_02';
:param batchSize => 512;
:param threshold => 0.614;
:param maxDepth => 12;
:param timeoutSeconds => 29;
:param region => 'us-west';
:param epoch => 53;
:param version => '4.6.4';

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_000' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.9936,
  latency: 162,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 7869,
  confidence: 0.1031,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_001' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.2621,
  latency: 210,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 4057,
  confidence: 0.3496,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_002' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.6311,
  latency: 246,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 4183,
  confidence: 0.4664,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_003' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_004' })
MERGE (a)-[r_003:CALIBRATES {
  strength: 0.2041,
  latency: 41,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 935,
  confidence: 0.5561,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_004' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.228,
  latency: 2,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 8219,
  confidence: 0.6288,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_005' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_006' })
MERGE (a)-[r_005:DEPENDS_ON {
  strength: 0.009,
  latency: 128,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 6049,
  confidence: 0.1509,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_006' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.1142,
  latency: 181,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 7785,
  confidence: 0.9612,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_007' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.049,
  latency: 3,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 6741,
  confidence: 0.6024,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_008' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.9251,
  latency: 123,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 1448,
  confidence: 0.5596,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_009' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.5062,
  latency: 60,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 1039,
  confidence: 0.8967,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_010' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.0765,
  latency: 68,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 4524,
  confidence: 0.6212,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_011' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_012' })
MERGE (a)-[r_011:MONITORS {
  strength: 0.3829,
  latency: 52,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 8031,
  confidence: 0.1367,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_012' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_013' })
MERGE (a)-[r_012:PRODUCES {
  strength: 0.2957,
  latency: 71,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 265,
  confidence: 0.6829,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_013' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.3527,
  latency: 17,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 8067,
  confidence: 0.4168,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_014' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.9612,
  latency: 165,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 7527,
  confidence: 0.967,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_015' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.4496,
  latency: 188,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 1920,
  confidence: 0.4721,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_016' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.453,
  latency: 73,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 7874,
  confidence: 0.3044,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_017' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_018' })
MERGE (a)-[r_017:VALIDATES {
  strength: 0.187,
  latency: 26,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 5380,
  confidence: 0.2038,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_018' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.218,
  latency: 239,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 6988,
  confidence: 0.7086,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_019' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.1843,
  latency: 163,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 636,
  confidence: 0.0927,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_020' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.5288,
  latency: 225,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 1994,
  confidence: 0.6695,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_021' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.9148,
  latency: 184,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 4614,
  confidence: 0.3699,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_022' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_023' })
MERGE (a)-[r_022:PRODUCES {
  strength: 0.5672,
  latency: 227,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 2116,
  confidence: 0.4732,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_023' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.5655,
  latency: 138,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 3967,
  confidence: 0.1471,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_024' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_025' })
MERGE (a)-[r_024:ROUTES_TO {
  strength: 0.4225,
  latency: 40,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 291,
  confidence: 0.709,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_025' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.6478,
  latency: 126,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 661,
  confidence: 0.0911,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_026' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_027' })
MERGE (a)-[r_026:MONITORS {
  strength: 0.2263,
  latency: 211,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9189,
  confidence: 0.5439,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_027' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_028' })
MERGE (a)-[r_027:MONITORS {
  strength: 0.2099,
  latency: 144,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 2967,
  confidence: 0.5008,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_028' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.112,
  latency: 245,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 6459,
  confidence: 0.6233,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_029' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_03_config_managers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.965,
  latency: 238,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 2685,
  confidence: 0.5812,
  active: true
}]->(b);
