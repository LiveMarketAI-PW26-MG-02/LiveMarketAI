:param namespace => 'compression_01_01';
:param batchSize => 64;
:param threshold => 0.809;
:param maxDepth => 9;
:param timeoutSeconds => 75;
:param region => 'us-west';
:param epoch => 54;
:param version => '5.6.1';

CREATE (n_000:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_000',
  name: 'node_000',
  version: '1.8',
  status: 'stable',
  priority: 2,
  weight: 0.5288,
  score: 0.4744,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_001',
  name: 'node_001',
  version: '1.7',
  status: 'stable',
  priority: 8,
  weight: 0.7697,
  score: 0.9214,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_002',
  name: 'node_002',
  version: '4.0',
  status: 'active',
  priority: 6,
  weight: 0.9728,
  score: 0.1511,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_003',
  name: 'node_003',
  version: '4.4',
  status: 'recovered',
  priority: 9,
  weight: 0.5539,
  score: 0.9137,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_004',
  name: 'node_004',
  version: '5.3',
  status: 'completed',
  priority: 2,
  weight: 0.551,
  score: 0.4594,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_005',
  name: 'node_005',
  version: '3.2',
  status: 'stable',
  priority: 10,
  weight: 0.816,
  score: 0.4188,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_006',
  name: 'node_006',
  version: '2.2',
  status: 'recovered',
  priority: 2,
  weight: 0.3595,
  score: 0.9382,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_007',
  name: 'node_007',
  version: '3.3',
  status: 'failed',
  priority: 10,
  weight: 0.8396,
  score: 0.0137,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_008',
  name: 'node_008',
  version: '3.7',
  status: 'degraded',
  priority: 6,
  weight: 0.7256,
  score: 0.178,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_009',
  name: 'node_009',
  version: '4.7',
  status: 'recovered',
  priority: 4,
  weight: 0.727,
  score: 0.3214,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_010',
  name: 'node_010',
  version: '3.2',
  status: 'stable',
  priority: 5,
  weight: 0.9459,
  score: 0.4777,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_011',
  name: 'node_011',
  version: '3.3',
  status: 'pending',
  priority: 4,
  weight: 0.4381,
  score: 0.0526,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_012',
  name: 'node_012',
  version: '1.7',
  status: 'recovered',
  priority: 5,
  weight: 0.3434,
  score: 0.821,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_013',
  name: 'node_013',
  version: '5.3',
  status: 'active',
  priority: 8,
  weight: 0.7604,
  score: 0.6956,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_014',
  name: 'node_014',
  version: '1.0',
  status: 'stable',
  priority: 5,
  weight: 0.979,
  score: 0.7913,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_015',
  name: 'node_015',
  version: '2.5',
  status: 'recovered',
  priority: 1,
  weight: 0.6251,
  score: 0.3281,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_016',
  name: 'node_016',
  version: '3.0',
  status: 'failed',
  priority: 1,
  weight: 0.456,
  score: 0.6691,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_017',
  name: 'node_017',
  version: '2.5',
  status: 'active',
  priority: 3,
  weight: 0.4243,
  score: 0.5198,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_018',
  name: 'node_018',
  version: '5.2',
  status: 'stable',
  priority: 6,
  weight: 0.3954,
  score: 0.7364,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_019',
  name: 'node_019',
  version: '3.5',
  status: 'active',
  priority: 3,
  weight: 0.6927,
  score: 0.1303,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_020',
  name: 'node_020',
  version: '5.7',
  status: 'active',
  priority: 8,
  weight: 0.3973,
  score: 0.4895,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_021',
  name: 'node_021',
  version: '1.9',
  status: 'completed',
  priority: 2,
  weight: 0.6402,
  score: 0.7346,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_022',
  name: 'node_022',
  version: '1.4',
  status: 'stable',
  priority: 4,
  weight: 0.6155,
  score: 0.7039,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_023',
  name: 'node_023',
  version: '4.2',
  status: 'completed',
  priority: 3,
  weight: 0.6701,
  score: 0.114,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_024',
  name: 'node_024',
  version: '2.3',
  status: 'pending',
  priority: 4,
  weight: 0.536,
  score: 0.4492,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_025',
  name: 'node_025',
  version: '4.3',
  status: 'pending',
  priority: 1,
  weight: 0.7765,
  score: 0.6199,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_026',
  name: 'node_026',
  version: '4.3',
  status: 'pending',
  priority: 9,
  weight: 0.2145,
  score: 0.2562,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_027',
  name: 'node_027',
  version: '4.5',
  status: 'failed',
  priority: 2,
  weight: 0.3888,
  score: 0.8042,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_028',
  name: 'node_028',
  version: '1.1',
  status: 'failed',
  priority: 3,
  weight: 0.3754,
  score: 0.2016,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_029',
  name: 'node_029',
  version: '5.5',
  status: 'stable',
  priority: 10,
  weight: 0.8954,
  score: 0.0116,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_030',
  name: 'node_030',
  version: '1.5',
  status: 'recovered',
  priority: 10,
  weight: 0.5393,
  score: 0.0019,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_031',
  name: 'node_031',
  version: '4.4',
  status: 'active',
  priority: 7,
  weight: 0.8782,
  score: 0.5958,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_032',
  name: 'node_032',
  version: '5.3',
  status: 'failed',
  priority: 7,
  weight: 0.4658,
  score: 0.0253,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_033',
  name: 'node_033',
  version: '3.7',
  status: 'recovered',
  priority: 10,
  weight: 0.9119,
  score: 0.8962,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_034',
  name: 'node_034',
  version: '4.8',
  status: 'failed',
  priority: 9,
  weight: 0.1963,
  score: 0.9867,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_035',
  name: 'node_035',
  version: '3.2',
  status: 'stable',
  priority: 6,
  weight: 0.659,
  score: 0.1315,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_036',
  name: 'node_036',
  version: '5.1',
  status: 'failed',
  priority: 5,
  weight: 0.5947,
  score: 0.9699,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_037',
  name: 'node_037',
  version: '1.2',
  status: 'degraded',
  priority: 2,
  weight: 0.5431,
  score: 0.5602,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_038',
  name: 'node_038',
  version: '5.5',
  status: 'completed',
  priority: 5,
  weight: 0.6483,
  score: 0.5015,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_10_utility_helpers_1_039',
  name: 'node_039',
  version: '5.9',
  status: 'failed',
  priority: 2,
  weight: 0.6912,
  score: 0.9912,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: false
});
