:param namespace => 'checkpointloader_02_02';
:param batchSize => 128;
:param threshold => 0.204;
:param maxDepth => 12;
:param timeoutSeconds => 64;
:param region => 'us-west';
:param epoch => 1;
:param version => '4.5.5';

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_000' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_001' })
MERGE (a)-[r_000:MONITORS {
  strength: 0.6504,
  latency: 37,
  established: datetime(),
  channel: 'channel_0',
  priority: 6,
  bandwidth: 1699,
  confidence: 0.9422,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_001' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.1856,
  latency: 2,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 9133,
  confidence: 0.9048,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_002' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.8374,
  latency: 187,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 786,
  confidence: 0.3129,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_003' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_004' })
MERGE (a)-[r_003:PRODUCES {
  strength: 0.5759,
  latency: 178,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 8686,
  confidence: 0.0987,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_004' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_005' })
MERGE (a)-[r_004:PRODUCES {
  strength: 0.4988,
  latency: 90,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 1593,
  confidence: 0.3473,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_005' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.9114,
  latency: 138,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 3095,
  confidence: 0.107,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_006' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_007' })
MERGE (a)-[r_006:ROUTES_TO {
  strength: 0.2655,
  latency: 79,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 8945,
  confidence: 0.2603,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_007' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.7025,
  latency: 77,
  established: datetime(),
  channel: 'channel_7',
  priority: 4,
  bandwidth: 1248,
  confidence: 0.8971,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_008' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.7211,
  latency: 104,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 5677,
  confidence: 0.8261,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_009' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.875,
  latency: 205,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 8768,
  confidence: 0.308,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_010' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_011' })
MERGE (a)-[r_010:DEPENDS_ON {
  strength: 0.1131,
  latency: 87,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 5577,
  confidence: 0.1407,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_011' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.6368,
  latency: 150,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 7104,
  confidence: 0.2712,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_012' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.5506,
  latency: 6,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 1716,
  confidence: 0.7627,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_013' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.1769,
  latency: 234,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 2778,
  confidence: 0.9049,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_014' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_015' })
MERGE (a)-[r_014:PRODUCES {
  strength: 0.7016,
  latency: 170,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 8440,
  confidence: 0.405,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_015' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_016' })
MERGE (a)-[r_015:CALIBRATES {
  strength: 0.1184,
  latency: 230,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 3424,
  confidence: 0.8765,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_016' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_017' })
MERGE (a)-[r_016:DEPENDS_ON {
  strength: 0.9817,
  latency: 127,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 7638,
  confidence: 0.1549,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_017' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.1085,
  latency: 188,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 7532,
  confidence: 0.7977,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_018' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.9236,
  latency: 205,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 9899,
  confidence: 0.3162,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_019' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.01,
  latency: 215,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 1622,
  confidence: 0.2037,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_020' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_021' })
MERGE (a)-[r_020:OBSERVES {
  strength: 0.0869,
  latency: 125,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 9654,
  confidence: 0.0972,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_021' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_022' })
MERGE (a)-[r_021:DEPENDS_ON {
  strength: 0.7254,
  latency: 118,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 6082,
  confidence: 0.1352,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_022' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.5941,
  latency: 249,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 6977,
  confidence: 0.7818,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_023' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.8745,
  latency: 112,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 9155,
  confidence: 0.9423,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_024' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_025' })
MERGE (a)-[r_024:VALIDATES {
  strength: 0.5018,
  latency: 168,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 5375,
  confidence: 0.5042,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_025' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_026' })
MERGE (a)-[r_025:OBSERVES {
  strength: 0.0321,
  latency: 88,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 6195,
  confidence: 0.6095,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_026' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.5407,
  latency: 35,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 5576,
  confidence: 0.3984,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_027' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_028' })
MERGE (a)-[r_027:ROUTES_TO {
  strength: 0.5986,
  latency: 86,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 6362,
  confidence: 0.4464,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_028' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_029' })
MERGE (a)-[r_028:PRODUCES {
  strength: 0.8147,
  latency: 57,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 2908,
  confidence: 0.1842,
  active: true
}]->(b);

MATCH (a:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_029' }),
      (b:CheckpointLoader { identifier: 'checkpointloader_02_state_handlers_2_000' })
MERGE (a)-[r_029:CALIBRATES {
  strength: 0.6214,
  latency: 101,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 1290,
  confidence: 0.8477,
  active: true
}]->(b);
