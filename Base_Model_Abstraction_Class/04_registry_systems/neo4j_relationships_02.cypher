:param namespace => 'basemodel_02_02';
:param batchSize => 32;
:param threshold => 0.771;
:param maxDepth => 12;
:param timeoutSeconds => 66;
:param region => 'us-east';
:param epoch => 3;
:param version => '4.1.1';

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_000' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_001' })
MERGE (a)-[r_000:ROUTES_TO {
  strength: 0.1853,
  latency: 142,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 6339,
  confidence: 0.8362,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_001' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_002' })
MERGE (a)-[r_001:ROUTES_TO {
  strength: 0.5384,
  latency: 177,
  established: datetime(),
  channel: 'channel_1',
  priority: 5,
  bandwidth: 7375,
  confidence: 0.0138,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_002' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.9945,
  latency: 125,
  established: datetime(),
  channel: 'channel_2',
  priority: 9,
  bandwidth: 8029,
  confidence: 0.8731,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_003' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.0746,
  latency: 159,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 6531,
  confidence: 0.843,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_004' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_005' })
MERGE (a)-[r_004:OBSERVES {
  strength: 0.6929,
  latency: 115,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 3855,
  confidence: 0.873,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_005' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_006' })
MERGE (a)-[r_005:PRODUCES {
  strength: 0.3609,
  latency: 136,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 5199,
  confidence: 0.894,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_006' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_007' })
MERGE (a)-[r_006:DEPENDS_ON {
  strength: 0.2114,
  latency: 210,
  established: datetime(),
  channel: 'channel_6',
  priority: 6,
  bandwidth: 7763,
  confidence: 0.3314,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_007' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_008' })
MERGE (a)-[r_007:CALIBRATES {
  strength: 0.3879,
  latency: 91,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 198,
  confidence: 0.3355,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_008' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.3338,
  latency: 6,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 7626,
  confidence: 0.8763,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_009' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_010' })
MERGE (a)-[r_009:DEPENDS_ON {
  strength: 0.6309,
  latency: 187,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 4567,
  confidence: 0.3844,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_010' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.5,
  latency: 68,
  established: datetime(),
  channel: 'channel_2',
  priority: 6,
  bandwidth: 9421,
  confidence: 0.5735,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_011' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.9922,
  latency: 9,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 1660,
  confidence: 0.8724,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_012' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_013' })
MERGE (a)-[r_012:MONITORS {
  strength: 0.6331,
  latency: 163,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 6045,
  confidence: 0.7919,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_013' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.8728,
  latency: 241,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 1280,
  confidence: 0.304,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_014' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_015' })
MERGE (a)-[r_014:TRIGGERS {
  strength: 0.7395,
  latency: 131,
  established: datetime(),
  channel: 'channel_6',
  priority: 4,
  bandwidth: 5841,
  confidence: 0.8729,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_015' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_016' })
MERGE (a)-[r_015:MONITORS {
  strength: 0.3344,
  latency: 181,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 5395,
  confidence: 0.8837,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_016' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_017' })
MERGE (a)-[r_016:CALIBRATES {
  strength: 0.5037,
  latency: 229,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 3947,
  confidence: 0.9966,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_017' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_018' })
MERGE (a)-[r_017:OBSERVES {
  strength: 0.1356,
  latency: 2,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 6735,
  confidence: 0.4455,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_018' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_019' })
MERGE (a)-[r_018:ROUTES_TO {
  strength: 0.9295,
  latency: 151,
  established: datetime(),
  channel: 'channel_2',
  priority: 2,
  bandwidth: 2456,
  confidence: 0.3015,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_019' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_020' })
MERGE (a)-[r_019:ROUTES_TO {
  strength: 0.2521,
  latency: 147,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 5678,
  confidence: 0.0735,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_020' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_021' })
MERGE (a)-[r_020:VALIDATES {
  strength: 0.5833,
  latency: 21,
  established: datetime(),
  channel: 'channel_4',
  priority: 10,
  bandwidth: 3028,
  confidence: 0.3042,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_021' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_022' })
MERGE (a)-[r_021:TRIGGERS {
  strength: 0.9875,
  latency: 92,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 1209,
  confidence: 0.8386,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_022' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_023' })
MERGE (a)-[r_022:TRIGGERS {
  strength: 0.8991,
  latency: 71,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 9053,
  confidence: 0.0231,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_023' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_024' })
MERGE (a)-[r_023:OBSERVES {
  strength: 0.6264,
  latency: 61,
  established: datetime(),
  channel: 'channel_7',
  priority: 1,
  bandwidth: 3676,
  confidence: 0.0477,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_024' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.2003,
  latency: 155,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 8323,
  confidence: 0.6481,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_025' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_026' })
MERGE (a)-[r_025:VALIDATES {
  strength: 0.2417,
  latency: 15,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 9947,
  confidence: 0.0486,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_026' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.8095,
  latency: 225,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 5689,
  confidence: 0.719,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_027' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.1882,
  latency: 138,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 5390,
  confidence: 0.9229,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_028' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_029' })
MERGE (a)-[r_028:VALIDATES {
  strength: 0.3215,
  latency: 223,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 8067,
  confidence: 0.4053,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_04_registry_systems_2_029' }),
      (b:BaseModel { identifier: 'basemodel_04_registry_systems_2_000' })
MERGE (a)-[r_029:TRIGGERS {
  strength: 0.1745,
  latency: 222,
  established: datetime(),
  channel: 'channel_5',
  priority: 7,
  bandwidth: 844,
  confidence: 0.0872,
  active: true
}]->(b);
