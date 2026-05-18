:param namespace => 'basemodel_02_02';
:param batchSize => 32;
:param threshold => 0.388;
:param maxDepth => 10;
:param timeoutSeconds => 39;
:param region => 'us-east';
:param epoch => 62;
:param version => '3.8.7';

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_000' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.6212,
  latency: 56,
  established: datetime(),
  channel: 'channel_0',
  priority: 4,
  bandwidth: 7807,
  confidence: 0.2019,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_001' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_002' })
MERGE (a)-[r_001:CALIBRATES {
  strength: 0.271,
  latency: 250,
  established: datetime(),
  channel: 'channel_1',
  priority: 6,
  bandwidth: 620,
  confidence: 0.407,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_002' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_003' })
MERGE (a)-[r_002:TRIGGERS {
  strength: 0.4131,
  latency: 182,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 9415,
  confidence: 0.3739,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_003' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_004' })
MERGE (a)-[r_003:OBSERVES {
  strength: 0.2384,
  latency: 215,
  established: datetime(),
  channel: 'channel_3',
  priority: 1,
  bandwidth: 2636,
  confidence: 0.6075,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_004' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_005' })
MERGE (a)-[r_004:ROUTES_TO {
  strength: 0.6066,
  latency: 122,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 9076,
  confidence: 0.7117,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_005' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_006' })
MERGE (a)-[r_005:OBSERVES {
  strength: 0.2611,
  latency: 144,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 4587,
  confidence: 0.9603,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_006' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_007' })
MERGE (a)-[r_006:OBSERVES {
  strength: 0.9104,
  latency: 134,
  established: datetime(),
  channel: 'channel_6',
  priority: 3,
  bandwidth: 9625,
  confidence: 0.3212,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_007' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_008' })
MERGE (a)-[r_007:DEPENDS_ON {
  strength: 0.1677,
  latency: 109,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 1414,
  confidence: 0.5856,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_008' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_009' })
MERGE (a)-[r_008:CALIBRATES {
  strength: 0.79,
  latency: 65,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 3753,
  confidence: 0.8602,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_009' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_010' })
MERGE (a)-[r_009:ROUTES_TO {
  strength: 0.9678,
  latency: 183,
  established: datetime(),
  channel: 'channel_1',
  priority: 7,
  bandwidth: 1653,
  confidence: 0.0516,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_010' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_011' })
MERGE (a)-[r_010:PRODUCES {
  strength: 0.9675,
  latency: 232,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 1255,
  confidence: 0.289,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_011' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.8705,
  latency: 108,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 8773,
  confidence: 0.3768,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_012' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_013' })
MERGE (a)-[r_012:ROUTES_TO {
  strength: 0.807,
  latency: 168,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 9653,
  confidence: 0.1166,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_013' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_014' })
MERGE (a)-[r_013:VALIDATES {
  strength: 0.4996,
  latency: 136,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 6154,
  confidence: 0.8988,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_014' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_015' })
MERGE (a)-[r_014:VALIDATES {
  strength: 0.436,
  latency: 152,
  established: datetime(),
  channel: 'channel_6',
  priority: 5,
  bandwidth: 9444,
  confidence: 0.382,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_015' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.6435,
  latency: 106,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 8682,
  confidence: 0.2574,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_016' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_017' })
MERGE (a)-[r_016:PRODUCES {
  strength: 0.701,
  latency: 15,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 7828,
  confidence: 0.2123,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_017' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.7996,
  latency: 3,
  established: datetime(),
  channel: 'channel_1',
  priority: 8,
  bandwidth: 7888,
  confidence: 0.34,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_018' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_019' })
MERGE (a)-[r_018:OBSERVES {
  strength: 0.4655,
  latency: 84,
  established: datetime(),
  channel: 'channel_2',
  priority: 4,
  bandwidth: 7155,
  confidence: 0.089,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_019' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_020' })
MERGE (a)-[r_019:VALIDATES {
  strength: 0.5425,
  latency: 103,
  established: datetime(),
  channel: 'channel_3',
  priority: 3,
  bandwidth: 3909,
  confidence: 0.3708,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_020' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_021' })
MERGE (a)-[r_020:TRIGGERS {
  strength: 0.3801,
  latency: 127,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 2190,
  confidence: 0.9977,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_021' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.8789,
  latency: 29,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 8453,
  confidence: 0.136,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_022' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_023' })
MERGE (a)-[r_022:MONITORS {
  strength: 0.616,
  latency: 166,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 7793,
  confidence: 0.5824,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_023' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_024' })
MERGE (a)-[r_023:TRIGGERS {
  strength: 0.577,
  latency: 92,
  established: datetime(),
  channel: 'channel_7',
  priority: 6,
  bandwidth: 7263,
  confidence: 0.3145,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_024' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.6931,
  latency: 174,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 6555,
  confidence: 0.3697,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_025' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_026' })
MERGE (a)-[r_025:ROUTES_TO {
  strength: 0.8355,
  latency: 165,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 4172,
  confidence: 0.7049,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_026' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_027' })
MERGE (a)-[r_026:VALIDATES {
  strength: 0.3692,
  latency: 218,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4290,
  confidence: 0.1634,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_027' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.6011,
  latency: 218,
  established: datetime(),
  channel: 'channel_3',
  priority: 10,
  bandwidth: 847,
  confidence: 0.1983,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_028' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.5955,
  latency: 106,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 4563,
  confidence: 0.0291,
  active: true
}]->(b);

MATCH (a:BaseModel { identifier: 'basemodel_06_validation_layer_2_029' }),
      (b:BaseModel { identifier: 'basemodel_06_validation_layer_2_000' })
MERGE (a)-[r_029:DEPENDS_ON {
  strength: 0.8371,
  latency: 22,
  established: datetime(),
  channel: 'channel_5',
  priority: 4,
  bandwidth: 164,
  confidence: 0.1736,
  active: true
}]->(b);
