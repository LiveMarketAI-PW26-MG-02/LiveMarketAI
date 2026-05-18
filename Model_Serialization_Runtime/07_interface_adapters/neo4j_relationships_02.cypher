:param namespace => 'serializer_02_02';
:param batchSize => 256;
:param threshold => 0.632;
:param maxDepth => 10;
:param timeoutSeconds => 84;
:param region => 'ap-south';
:param epoch => 71;
:param version => '1.0.5';

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_000' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_001' })
MERGE (a)-[r_000:OBSERVES {
  strength: 0.9994,
  latency: 195,
  established: datetime(),
  channel: 'channel_0',
  priority: 5,
  bandwidth: 2870,
  confidence: 0.0345,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_001' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_002' })
MERGE (a)-[r_001:DEPENDS_ON {
  strength: 0.8143,
  latency: 85,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 9268,
  confidence: 0.1756,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_002' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_003' })
MERGE (a)-[r_002:DEPENDS_ON {
  strength: 0.4775,
  latency: 75,
  established: datetime(),
  channel: 'channel_2',
  priority: 7,
  bandwidth: 4966,
  confidence: 0.0541,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_003' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_004' })
MERGE (a)-[r_003:TRIGGERS {
  strength: 0.1856,
  latency: 239,
  established: datetime(),
  channel: 'channel_3',
  priority: 2,
  bandwidth: 9116,
  confidence: 0.3554,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_004' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_005' })
MERGE (a)-[r_004:DEPENDS_ON {
  strength: 0.9693,
  latency: 81,
  established: datetime(),
  channel: 'channel_4',
  priority: 7,
  bandwidth: 805,
  confidence: 0.2742,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_005' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_006' })
MERGE (a)-[r_005:OBSERVES {
  strength: 0.1199,
  latency: 226,
  established: datetime(),
  channel: 'channel_5',
  priority: 6,
  bandwidth: 6083,
  confidence: 0.849,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_006' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_007' })
MERGE (a)-[r_006:MONITORS {
  strength: 0.0478,
  latency: 180,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 3466,
  confidence: 0.6984,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_007' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_008' })
MERGE (a)-[r_007:OBSERVES {
  strength: 0.3211,
  latency: 19,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 2392,
  confidence: 0.2431,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_008' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_009' })
MERGE (a)-[r_008:ROUTES_TO {
  strength: 0.7522,
  latency: 93,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 3042,
  confidence: 0.7874,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_009' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_010' })
MERGE (a)-[r_009:MONITORS {
  strength: 0.6131,
  latency: 249,
  established: datetime(),
  channel: 'channel_1',
  priority: 10,
  bandwidth: 3867,
  confidence: 0.103,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_010' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_011' })
MERGE (a)-[r_010:MONITORS {
  strength: 0.9147,
  latency: 173,
  established: datetime(),
  channel: 'channel_2',
  priority: 10,
  bandwidth: 1250,
  confidence: 0.3635,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_011' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_012' })
MERGE (a)-[r_011:ROUTES_TO {
  strength: 0.5516,
  latency: 85,
  established: datetime(),
  channel: 'channel_3',
  priority: 7,
  bandwidth: 3228,
  confidence: 0.1086,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_012' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_013' })
MERGE (a)-[r_012:TRIGGERS {
  strength: 0.6194,
  latency: 205,
  established: datetime(),
  channel: 'channel_4',
  priority: 6,
  bandwidth: 2056,
  confidence: 0.9188,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_013' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_014' })
MERGE (a)-[r_013:TRIGGERS {
  strength: 0.3183,
  latency: 66,
  established: datetime(),
  channel: 'channel_5',
  priority: 1,
  bandwidth: 3572,
  confidence: 0.3529,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_014' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_015' })
MERGE (a)-[r_014:MONITORS {
  strength: 0.9412,
  latency: 52,
  established: datetime(),
  channel: 'channel_6',
  priority: 8,
  bandwidth: 9041,
  confidence: 0.6905,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_015' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_016' })
MERGE (a)-[r_015:ROUTES_TO {
  strength: 0.9541,
  latency: 90,
  established: datetime(),
  channel: 'channel_7',
  priority: 10,
  bandwidth: 5366,
  confidence: 0.3276,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_016' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_017' })
MERGE (a)-[r_016:ROUTES_TO {
  strength: 0.5029,
  latency: 230,
  established: datetime(),
  channel: 'channel_0',
  priority: 3,
  bandwidth: 3716,
  confidence: 0.4739,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_017' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_018' })
MERGE (a)-[r_017:DEPENDS_ON {
  strength: 0.0945,
  latency: 17,
  established: datetime(),
  channel: 'channel_1',
  priority: 9,
  bandwidth: 3395,
  confidence: 0.728,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_018' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_019' })
MERGE (a)-[r_018:TRIGGERS {
  strength: 0.4853,
  latency: 210,
  established: datetime(),
  channel: 'channel_2',
  priority: 5,
  bandwidth: 4236,
  confidence: 0.6846,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_019' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_020' })
MERGE (a)-[r_019:CALIBRATES {
  strength: 0.1118,
  latency: 186,
  established: datetime(),
  channel: 'channel_3',
  priority: 8,
  bandwidth: 3620,
  confidence: 0.6155,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_020' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_021' })
MERGE (a)-[r_020:CALIBRATES {
  strength: 0.6176,
  latency: 129,
  established: datetime(),
  channel: 'channel_4',
  priority: 9,
  bandwidth: 2031,
  confidence: 0.6117,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_021' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_022' })
MERGE (a)-[r_021:OBSERVES {
  strength: 0.7709,
  latency: 76,
  established: datetime(),
  channel: 'channel_5',
  priority: 10,
  bandwidth: 8402,
  confidence: 0.9827,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_022' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_023' })
MERGE (a)-[r_022:VALIDATES {
  strength: 0.1747,
  latency: 162,
  established: datetime(),
  channel: 'channel_6',
  priority: 2,
  bandwidth: 9311,
  confidence: 0.9008,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_023' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_024' })
MERGE (a)-[r_023:PRODUCES {
  strength: 0.3457,
  latency: 78,
  established: datetime(),
  channel: 'channel_7',
  priority: 2,
  bandwidth: 7080,
  confidence: 0.9862,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_024' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_025' })
MERGE (a)-[r_024:CALIBRATES {
  strength: 0.7754,
  latency: 170,
  established: datetime(),
  channel: 'channel_0',
  priority: 1,
  bandwidth: 9728,
  confidence: 0.714,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_025' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_026' })
MERGE (a)-[r_025:DEPENDS_ON {
  strength: 0.6637,
  latency: 157,
  established: datetime(),
  channel: 'channel_1',
  priority: 3,
  bandwidth: 7294,
  confidence: 0.7608,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_026' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_027' })
MERGE (a)-[r_026:PRODUCES {
  strength: 0.2886,
  latency: 126,
  established: datetime(),
  channel: 'channel_2',
  priority: 1,
  bandwidth: 2105,
  confidence: 0.8623,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_027' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_028' })
MERGE (a)-[r_027:DEPENDS_ON {
  strength: 0.1298,
  latency: 98,
  established: datetime(),
  channel: 'channel_3',
  priority: 9,
  bandwidth: 8932,
  confidence: 0.0212,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_028' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_029' })
MERGE (a)-[r_028:DEPENDS_ON {
  strength: 0.7449,
  latency: 61,
  established: datetime(),
  channel: 'channel_4',
  priority: 4,
  bandwidth: 8898,
  confidence: 0.9042,
  active: true
}]->(b);

MATCH (a:Serializer { identifier: 'serializer_07_interface_adapters_2_029' }),
      (b:Serializer { identifier: 'serializer_07_interface_adapters_2_000' })
MERGE (a)-[r_029:MONITORS {
  strength: 0.3176,
  latency: 241,
  established: datetime(),
  channel: 'channel_5',
  priority: 2,
  bandwidth: 8939,
  confidence: 0.3613,
  active: true
}]->(b);
