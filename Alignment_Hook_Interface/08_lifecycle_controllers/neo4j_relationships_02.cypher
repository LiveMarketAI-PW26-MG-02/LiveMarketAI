:param namespace => 'alignment_02_02';
:param batchSize => 128;
:param threshold => 0.73;
:param maxDepth => 7;
:param timeoutSeconds => 80;
:param region => 'us-east';
:param epoch => 82;
:param version => '5.9.3';

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_000' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_001' })
MERGE (a)-[r_000:DEPENDS_ON {
  strength: 0.7829,
  latency: 183,
  established: datetime(),
  channel: 'channel_0',
  priority: 7,
  bandwidth: 1753,
  confidence: 0.6864,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_001' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_002' })
MERGE (a)-[r_001:TRIGGERS {
  strength: 0.8784,
  latency: 229,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 8559,
  confidence: 0.1812,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_002' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_003' })
MERGE (a)-[r_002:ROUTES_TO {
  strength: 0.9815,
  latency: 15,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 5568,
  confidence: 0.9361,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_003' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.8261,
  latency: 240,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 1285,
  confidence: 0.274,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_004' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_005' })
MERGE (a)-[r_004:VALIDATES {
  strength: 0.7833,
  latency: 168,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 9432,
  confidence: 0.9208,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_005' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_006' })
MERGE (a)-[r_005:VALIDATES {
  strength: 0.8111,
  latency: 14,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 678,
  confidence: 0.1665,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_006' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.2833,
  latency: 11,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 4459,
  confidence: 0.4548,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_007' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_008' })
MERGE (a)-[r_007:VALIDATES {
  strength: 0.7776,
  latency: 238,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 7423,
  confidence: 0.1688,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_008' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_009' })
MERGE (a)-[r_008:MONITORS {
  strength: 0.5021,
  latency: 1,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 9435,
  confidence: 0.2456,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_009' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_010' })
MERGE (a)-[r_009:CALIBRATES {
  strength: 0.0913,
  latency: 44,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 7210,
  confidence: 0.2913,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_010' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_011' })
MERGE (a)-[r_010:OBSERVES {
  strength: 0.9925,
  latency: 233,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 9083,
  confidence: 0.8973,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_011' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_012' })
MERGE (a)-[r_011:PRODUCES {
  strength: 0.0683,
  latency: 71,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 8698,
  confidence: 0.1865,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_012' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.4966,
  latency: 34,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 9034,
  confidence: 0.4502,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_013' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_014' })
MERGE (a)-[r_013:ROUTES_TO {
  strength: 0.7438,
  latency: 27,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 776,
  confidence: 0.8818,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_014' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_015' })
MERGE (a)-[r_014:DEPENDS_ON {
  strength: 0.3503,
  latency: 3,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 3653,
  confidence: 0.5443,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_015' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_016' })
MERGE (a)-[r_015:TRIGGERS {
  strength: 0.8238,
  latency: 233,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 3847,
  confidence: 0.3162,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_016' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.3938,
  latency: 217,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 8582,
  confidence: 0.4685,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_017' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_018' })
MERGE (a)-[r_017:MONITORS {
  strength: 0.7889,
  latency: 180,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 2055,
  confidence: 0.4584,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_018' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.5027,
  latency: 109,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 1883,
  confidence: 0.9434,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_019' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.6096,
  latency: 153,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 3734,
  confidence: 0.3284,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_020' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_021' })
MERGE (a)-[r_020:ROUTES_TO {
  strength: 0.086,
  latency: 55,
  established: datetime(),
  channel: 'channel_4',
  priority: 3,
  bandwidth: 2901,
  confidence: 0.6088,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_021' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_022' })
MERGE (a)-[r_021:PRODUCES {
  strength: 0.264,
  latency: 11,
  established: datetime(),
  channel: 'channel_5',
  priority: 9,
  bandwidth: 1782,
  confidence: 0.6919,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_022' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_023' })
MERGE (a)-[r_022:OBSERVES {
  strength: 0.9382,
  latency: 9,
  established: datetime(),
  channel: 'channel_6',
  priority: 1,
  bandwidth: 926,
  confidence: 0.4612,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_023' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_024' })
MERGE (a)-[r_023:VALIDATES {
  strength: 0.2675,
  latency: 227,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 3374,
  confidence: 0.5554,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_024' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_025' })
MERGE (a)-[r_024:OBSERVES {
  strength: 0.6087,
  latency: 126,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 1172,
  confidence: 0.0501,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_025' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.754,
  latency: 188,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 6194,
  confidence: 0.8746,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_026' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.0303,
  latency: 165,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9204,
  confidence: 0.2568,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_027' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.2159,
  latency: 132,
  established: datetime(),
  channel: 'channel_3',
  priority: 6,
  bandwidth: 3616,
  confidence: 0.6922,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_028' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_029' })
MERGE (a)-[r_028:TRIGGERS {
  strength: 0.7539,
  latency: 55,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 7366,
  confidence: 0.9567,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_029' }),
      (b:Alignment { identifier: 'alignment_08_lifecycle_controllers_2_000' })
MERGE (a)-[r_029:PRODUCES {
  strength: 0.9571,
  latency: 31,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 618,
  confidence: 0.6869,
  active: true
}]->(b);
