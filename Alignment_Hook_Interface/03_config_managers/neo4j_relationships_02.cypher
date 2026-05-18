:param namespace => 'alignment_02_02';
:param batchSize => 32;
:param threshold => 0.404;
:param maxDepth => 3;
:param timeoutSeconds => 55;
:param region => 'us-east';
:param epoch => 27;
:param version => '2.8.9';

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_000' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.321,
  latency: 57,
  established: datetime(),
  channel: 'channel_0',
  priority: 9,
  bandwidth: 174,
  confidence: 0.6802,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_001' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_002' })
MERGE (a)-[r_001:PRODUCES {
  strength: 0.5665,
  latency: 4,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 1999,
  confidence: 0.3203,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_002' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.226,
  latency: 59,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 9526,
  confidence: 0.5273,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_003' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_004' })
MERGE (a)-[r_003:MONITORS {
  strength: 0.2939,
  latency: 141,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 3829,
  confidence: 0.8773,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_004' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.0121,
  latency: 170,
  established: datetime(),
  channel: 'channel_4',
  priority: 5,
  bandwidth: 4921,
  confidence: 0.6937,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_005' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_006' })
MERGE (a)-[r_005:ROUTES_TO {
  strength: 0.6229,
  latency: 151,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 3708,
  confidence: 0.9588,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_006' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.831,
  latency: 157,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 8239,
  confidence: 0.042,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_007' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.9946,
  latency: 48,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 6485,
  confidence: 0.5427,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_008' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.5454,
  latency: 95,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 8857,
  confidence: 0.9186,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_009' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.8441,
  latency: 46,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 1842,
  confidence: 0.6777,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_010' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_011' })
MERGE (a)-[r_010:CALIBRATES {
  strength: 0.6716,
  latency: 167,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 575,
  confidence: 0.9244,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_011' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_012' })
MERGE (a)-[r_011:CALIBRATES {
  strength: 0.9951,
  latency: 152,
  established: datetime(),
  channel: 'channel_3',
  priority: 4,
  bandwidth: 9746,
  confidence: 0.0925,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_012' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_013' })
MERGE (a)-[r_012:DEPENDS_ON {
  strength: 0.7292,
  latency: 38,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6078,
  confidence: 0.8473,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_013' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.7736,
  latency: 56,
  established: datetime(),
  channel: 'channel_5',
  priority: 5,
  bandwidth: 4407,
  confidence: 0.5071,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_014' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_015' })
MERGE (a)-[r_014:CALIBRATES {
  strength: 0.8539,
  latency: 41,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 5524,
  confidence: 0.0917,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_015' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.1192,
  latency: 148,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 526,
  confidence: 0.3942,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_016' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_017' })
MERGE (a)-[r_016:OBSERVES {
  strength: 0.5972,
  latency: 4,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 760,
  confidence: 0.2321,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_017' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_018' })
MERGE (a)-[r_017:CALIBRATES {
  strength: 0.8042,
  latency: 21,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 6844,
  confidence: 0.7185,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_018' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_019' })
MERGE (a)-[r_018:PRODUCES {
  strength: 0.9871,
  latency: 60,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 7327,
  confidence: 0.4913,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_019' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_020' })
MERGE (a)-[r_019:MONITORS {
  strength: 0.1401,
  latency: 246,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 9115,
  confidence: 0.9915,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_020' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.6969,
  latency: 168,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 7257,
  confidence: 0.5872,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_021' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.3615,
  latency: 171,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 5620,
  confidence: 0.7027,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_022' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_023' })
MERGE (a)-[r_022:ROUTES_TO {
  strength: 0.8005,
  latency: 228,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 4698,
  confidence: 0.3734,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_023' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.7711,
  latency: 61,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 470,
  confidence: 0.6123,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_024' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_025' })
MERGE (a)-[r_024:MONITORS {
  strength: 0.6788,
  latency: 18,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 2795,
  confidence: 0.535,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_025' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.9464,
  latency: 18,
  established: datetime(),
  channel: 'channel_1',
  priority: 2,
  bandwidth: 7458,
  confidence: 0.2887,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_026' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_027' })
MERGE (a)-[r_026:DEPENDS_ON {
  strength: 0.2118,
  latency: 129,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4939,
  confidence: 0.7461,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_027' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.2067,
  latency: 49,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 319,
  confidence: 0.9381,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_028' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_029' })
MERGE (a)-[r_028:ROUTES_TO {
  strength: 0.9605,
  latency: 171,
  established: datetime(),
  channel: 'channel_4',
  priority: 8,
  bandwidth: 7606,
  confidence: 0.5621,
  active: true
}]->(b);

MATCH (a:Alignment { identifier: 'alignment_03_config_managers_2_029' }),
      (b:Alignment { identifier: 'alignment_03_config_managers_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.445,
  latency: 53,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 6952,
  confidence: 0.8814,
  active: true
}]->(b);
