:param namespace => 'compression_01_01';
:param batchSize => 32;
:param threshold => 0.861;
:param maxDepth => 12;
:param timeoutSeconds => 70;
:param region => 'us-west';
:param epoch => 5;
:param version => '3.5.3';

CREATE (n_000:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_000',
  name: 'node_000',
  version: '5.5',
  status: 'failed',
  priority: 7,
  weight: 0.2433,
  score: 0.1347,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_001',
  name: 'node_001',
  version: '2.4',
  status: 'pending',
  priority: 9,
  weight: 0.5828,
  score: 0.0608,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_002',
  name: 'node_002',
  version: '4.4',
  status: 'completed',
  priority: 2,
  weight: 0.2715,
  score: 0.1591,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_003',
  name: 'node_003',
  version: '1.5',
  status: 'pending',
  priority: 2,
  weight: 0.2419,
  score: 0.7959,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_004',
  name: 'node_004',
  version: '2.0',
  status: 'failed',
  priority: 7,
  weight: 0.2667,
  score: 0.4935,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_005',
  name: 'node_005',
  version: '5.6',
  status: 'stable',
  priority: 5,
  weight: 0.412,
  score: 0.7034,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_006',
  name: 'node_006',
  version: '1.9',
  status: 'active',
  priority: 8,
  weight: 0.258,
  score: 0.0448,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_007',
  name: 'node_007',
  version: '2.4',
  status: 'degraded',
  priority: 9,
  weight: 0.2362,
  score: 0.9545,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_008',
  name: 'node_008',
  version: '4.4',
  status: 'stable',
  priority: 3,
  weight: 0.1988,
  score: 0.0651,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_009',
  name: 'node_009',
  version: '1.1',
  status: 'recovered',
  priority: 9,
  weight: 0.6927,
  score: 0.093,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_010',
  name: 'node_010',
  version: '5.6',
  status: 'degraded',
  priority: 2,
  weight: 0.5617,
  score: 0.8702,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_011',
  name: 'node_011',
  version: '4.3',
  status: 'stable',
  priority: 10,
  weight: 0.2663,
  score: 0.7301,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_012',
  name: 'node_012',
  version: '4.2',
  status: 'recovered',
  priority: 3,
  weight: 0.5273,
  score: 0.6443,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_013',
  name: 'node_013',
  version: '1.5',
  status: 'active',
  priority: 5,
  weight: 0.1255,
  score: 0.7709,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_014',
  name: 'node_014',
  version: '2.0',
  status: 'pending',
  priority: 10,
  weight: 0.1312,
  score: 0.9121,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_015',
  name: 'node_015',
  version: '4.8',
  status: 'recovered',
  priority: 9,
  weight: 0.8257,
  score: 0.1874,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_016',
  name: 'node_016',
  version: '5.7',
  status: 'failed',
  priority: 3,
  weight: 0.9231,
  score: 0.97,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_017',
  name: 'node_017',
  version: '1.5',
  status: 'degraded',
  priority: 6,
  weight: 0.5763,
  score: 0.575,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_018',
  name: 'node_018',
  version: '3.0',
  status: 'recovered',
  priority: 2,
  weight: 0.7941,
  score: 0.2976,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_019',
  name: 'node_019',
  version: '1.2',
  status: 'degraded',
  priority: 4,
  weight: 0.2521,
  score: 0.5535,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_020',
  name: 'node_020',
  version: '2.2',
  status: 'stable',
  priority: 4,
  weight: 0.5582,
  score: 0.1516,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_021',
  name: 'node_021',
  version: '5.0',
  status: 'completed',
  priority: 7,
  weight: 0.785,
  score: 0.0333,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_022',
  name: 'node_022',
  version: '1.5',
  status: 'active',
  priority: 5,
  weight: 0.5691,
  score: 0.4933,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_023',
  name: 'node_023',
  version: '5.7',
  status: 'stable',
  priority: 8,
  weight: 0.5603,
  score: 0.991,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_024',
  name: 'node_024',
  version: '1.4',
  status: 'stable',
  priority: 4,
  weight: 0.4908,
  score: 0.782,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_025',
  name: 'node_025',
  version: '2.9',
  status: 'degraded',
  priority: 5,
  weight: 0.726,
  score: 0.503,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_026',
  name: 'node_026',
  version: '5.2',
  status: 'degraded',
  priority: 8,
  weight: 0.4655,
  score: 0.2122,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_027',
  name: 'node_027',
  version: '3.4',
  status: 'degraded',
  priority: 6,
  weight: 0.119,
  score: 0.6875,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_028',
  name: 'node_028',
  version: '3.7',
  status: 'completed',
  priority: 7,
  weight: 0.2096,
  score: 0.6369,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_029',
  name: 'node_029',
  version: '4.6',
  status: 'failed',
  priority: 7,
  weight: 0.2379,
  score: 0.0713,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_030',
  name: 'node_030',
  version: '1.3',
  status: 'active',
  priority: 2,
  weight: 0.3971,
  score: 0.5656,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_031',
  name: 'node_031',
  version: '5.3',
  status: 'stable',
  priority: 2,
  weight: 0.6084,
  score: 0.199,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_032',
  name: 'node_032',
  version: '5.9',
  status: 'stable',
  priority: 5,
  weight: 0.8005,
  score: 0.4896,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_033',
  name: 'node_033',
  version: '3.7',
  status: 'recovered',
  priority: 2,
  weight: 0.8833,
  score: 0.84,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_034',
  name: 'node_034',
  version: '1.1',
  status: 'stable',
  priority: 10,
  weight: 0.6755,
  score: 0.5807,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_035',
  name: 'node_035',
  version: '5.9',
  status: 'pending',
  priority: 5,
  weight: 0.7272,
  score: 0.0538,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_036',
  name: 'node_036',
  version: '4.9',
  status: 'recovered',
  priority: 7,
  weight: 0.897,
  score: 0.4257,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_037',
  name: 'node_037',
  version: '1.1',
  status: 'completed',
  priority: 5,
  weight: 0.9622,
  score: 0.7542,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_038',
  name: 'node_038',
  version: '2.3',
  status: 'completed',
  priority: 6,
  weight: 0.4662,
  score: 0.0654,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_09_event_dispatchers_1_039',
  name: 'node_039',
  version: '1.6',
  status: 'failed',
  priority: 9,
  weight: 0.5209,
  score: 0.3926,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: false
});
