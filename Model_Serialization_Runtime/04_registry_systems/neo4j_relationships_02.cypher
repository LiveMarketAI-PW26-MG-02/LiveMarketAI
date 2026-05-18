:param namespace => 'serializer_02_02';
:param batchSize => 32;
:param threshold => 0.514;
:param maxDepth => 3;
:param timeoutSeconds => 38;
:param region => 'ap-south';
:param epoch => 33;
:param version => '5.1.7';

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_000' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_001' })
MERGE (a)-[r_000:VALIDATES {
  strength: 0.1337,
  latency: 166,
  established: datetime(),
  channel: 'channel_0',
  priority: 10,
  bandwidth: 404,
  confidence: 0.315,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_001' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_002' })
MERGE (a)-[r_001:MONITORS {
  strength: 0.6737,
  latency: 126,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 1186,
  confidence: 0.9178,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_002' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_003' })
MERGE (a)-[r_002:VALIDATES {
  strength: 0.6304,
  latency: 131,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 7335,
  confidence: 0.4985,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_003' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_004' })
MERGE (a)-[r_003:DEPENDS_ON {
  strength: 0.5257,
  latency: 241,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 4129,
  confidence: 0.6112,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_004' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_005' })
MERGE (a)-[r_004:MONITORS {
  strength: 0.0271,
  latency: 12,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 1619,
  confidence: 0.1657,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_005' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_006' })
MERGE (a)-[r_005:CALIBRATES {
  strength: 0.437,
  latency: 125,
  established: datetime(),
  channel: 'channel_5',
  priority: 8,
  bandwidth: 2686,
  confidence: 0.0634,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_006' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_007' })
MERGE (a)-[r_006:VALIDATES {
  strength: 0.3023,
  latency: 44,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 2863,
  confidence: 0.5416,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_007' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_008' })
MERGE (a)-[r_007:MONITORS {
  strength: 0.7924,
  latency: 14,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 3547,
  confidence: 0.8875,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_008' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_009' })
MERGE (a)-[r_008:DEPENDS_ON {
  strength: 0.7613,
  latency: 54,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 775,
  confidence: 0.6935,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_009' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_010' })
MERGE (a)-[r_009:PRODUCES {
  strength: 0.7928,
  latency: 61,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 7761,
  confidence: 0.5646,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_010' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_011' })
MERGE (a)-[r_010:ROUTES_TO {
  strength: 0.5535,
  latency: 242,
  established: datetime(),
  channel: 'channel_2',
  priority: 8,
  bandwidth: 980,
  confidence: 0.4767,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_011' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_012' })
MERGE (a)-[r_011:OBSERVES {
  strength: 0.8796,
  latency: 90,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 7069,
  confidence: 0.055,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_012' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_013' })
MERGE (a)-[r_012:CALIBRATES {
  strength: 0.5236,
  latency: 47,
  established: datetime(),
  channel: 'channel_4',
  priority: 2,
  bandwidth: 1521,
  confidence: 0.7364,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_013' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_014' })
MERGE (a)-[r_013:OBSERVES {
  strength: 0.433,
  latency: 125,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 7692,
  confidence: 0.4027,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_014' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_015' })
MERGE (a)-[r_014:ROUTES_TO {
  strength: 0.4176,
  latency: 207,
  established: datetime(),
  channel: 'channel_6',
  priority: 7,
  bandwidth: 822,
  confidence: 0.4556,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_015' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_016' })
MERGE (a)-[r_015:PRODUCES {
  strength: 0.3561,
  latency: 142,
  established: datetime(),
  channel: 'channel_7',
  priority: 7,
  bandwidth: 7744,
  confidence: 0.2257,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_016' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_017' })
MERGE (a)-[r_016:PRODUCES {
  strength: 0.1892,
  latency: 45,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 7857,
  confidence: 0.8554,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_017' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_018' })
MERGE (a)-[r_017:TRIGGERS {
  strength: 0.6994,
  latency: 56,
  established: datetime(),
  channel: 'channel_1',
  priority: 1,
  bandwidth: 8913,
  confidence: 0.1317,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_018' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_019' })
MERGE (a)-[r_018:MONITORS {
  strength: 0.9853,
  latency: 27,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 2171,
  confidence: 0.1917,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_019' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_020' })
MERGE (a)-[r_019:OBSERVES {
  strength: 0.2375,
  latency: 247,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 5244,
  confidence: 0.4162,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_020' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.9637,
  latency: 107,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 2639,
  confidence: 0.8587,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_021' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_022' })
MERGE (a)-[r_021:VALIDATES {
  strength: 0.2448,
  latency: 112,
  established: datetime(),
  channel: 'channel_5',
  priority: 3,
  bandwidth: 3280,
  confidence: 0.6201,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_022' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.7363,
  latency: 77,
  established: datetime(),
  channel: 'channel_6',
  priority: 9,
  bandwidth: 6300,
  confidence: 0.9108,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_023' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_024' })
MERGE (a)-[r_023:DEPENDS_ON {
  strength: 0.8618,
  latency: 63,
  established: datetime(),
  channel: 'channel_7',
  priority: 3,
  bandwidth: 4074,
  confidence: 0.6782,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_024' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_025' })
MERGE (a)-[r_024:DEPENDS_ON {
  strength: 0.8084,
  latency: 178,
  established: datetime(),
  channel: 'channel_0',
  priority: 8,
  bandwidth: 5736,
  confidence: 0.8564,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_025' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.379,
  latency: 85,
  established: datetime(),
  channel: 'channel_1',
  priority: 4,
  bandwidth: 5019,
  confidence: 0.4289,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_026' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_027' })
MERGE (a)-[r_026:ROUTES_TO {
  strength: 0.5273,
  latency: 148,
  established: datetime(),
  channel: 'channel_2',
  priority: 3,
  bandwidth: 2767,
  confidence: 0.6155,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_027' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_028' })
MERGE (a)-[r_027:PRODUCES {
  strength: 0.2076,
  latency: 88,
  established: datetime(),
  channel: 'channel_3',
  priority: 5,
  bandwidth: 4434,
  confidence: 0.8953,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_028' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_029' })
MERGE (a)-[r_028:ROUTES_TO {
  strength: 0.523,
  latency: 121,
  established: datetime(),
  channel: 'channel_4',
  priority: 1,
  bandwidth: 4438,
  confidence: 0.4404,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_04_registry_systems_2_029' }),
      (b:Serializer { identifier: 'serializer_04_registry_systems_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.2853,
  latency: 151,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 8972,
  confidence: 0.2271,
  active: true
}]->(b);
