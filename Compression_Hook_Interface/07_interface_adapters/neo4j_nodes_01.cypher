:param namespace => 'compression_01_01';
:param batchSize => 512;
:param threshold => 0.862;
:param maxDepth => 9;
:param timeoutSeconds => 90;
:param region => 'us-east';
:param epoch => 67;
:param version => '4.2.9';

CREATE (n_000:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_000',
  name: 'node_000',
  version: '1.2',
  status: 'active',
  priority: 1,
  weight: 0.2238,
  score: 0.1901,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_001',
  name: 'node_001',
  version: '5.6',
  status: 'failed',
  priority: 9,
  weight: 0.3435,
  score: 0.1063,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_002',
  name: 'node_002',
  version: '2.2',
  status: 'stable',
  priority: 1,
  weight: 0.957,
  score: 0.4175,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_003',
  name: 'node_003',
  version: '4.1',
  status: 'failed',
  priority: 5,
  weight: 0.5851,
  score: 0.2582,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_004',
  name: 'node_004',
  version: '4.3',
  status: 'recovered',
  priority: 8,
  weight: 0.5687,
  score: 0.5481,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_005',
  name: 'node_005',
  version: '1.7',
  status: 'pending',
  priority: 1,
  weight: 0.2654,
  score: 0.5146,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_006',
  name: 'node_006',
  version: '2.5',
  status: 'completed',
  priority: 1,
  weight: 0.1741,
  score: 0.6141,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_007',
  name: 'node_007',
  version: '2.9',
  status: 'active',
  priority: 1,
  weight: 0.2115,
  score: 0.6207,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_008',
  name: 'node_008',
  version: '5.3',
  status: 'stable',
  priority: 7,
  weight: 0.6408,
  score: 0.9171,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_009',
  name: 'node_009',
  version: '4.2',
  status: 'failed',
  priority: 2,
  weight: 0.4991,
  score: 0.792,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_010',
  name: 'node_010',
  version: '1.7',
  status: 'degraded',
  priority: 2,
  weight: 0.9637,
  score: 0.7424,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_011',
  name: 'node_011',
  version: '5.8',
  status: 'degraded',
  priority: 5,
  weight: 0.5957,
  score: 0.2921,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_012',
  name: 'node_012',
  version: '5.2',
  status: 'failed',
  priority: 10,
  weight: 0.8416,
  score: 0.5866,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_013',
  name: 'node_013',
  version: '5.1',
  status: 'completed',
  priority: 7,
  weight: 0.8811,
  score: 0.5908,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_014',
  name: 'node_014',
  version: '4.9',
  status: 'stable',
  priority: 3,
  weight: 0.4521,
  score: 0.6971,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_015',
  name: 'node_015',
  version: '2.0',
  status: 'recovered',
  priority: 2,
  weight: 0.2164,
  score: 0.4342,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_016',
  name: 'node_016',
  version: '2.8',
  status: 'recovered',
  priority: 3,
  weight: 0.9115,
  score: 0.2989,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_017',
  name: 'node_017',
  version: '5.8',
  status: 'pending',
  priority: 5,
  weight: 0.1834,
  score: 0.2677,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_018',
  name: 'node_018',
  version: '3.5',
  status: 'pending',
  priority: 8,
  weight: 0.7376,
  score: 0.9991,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_019',
  name: 'node_019',
  version: '3.3',
  status: 'failed',
  priority: 10,
  weight: 0.3506,
  score: 0.7472,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_020',
  name: 'node_020',
  version: '2.6',
  status: 'pending',
  priority: 5,
  weight: 0.1813,
  score: 0.8208,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_021',
  name: 'node_021',
  version: '4.0',
  status: 'failed',
  priority: 3,
  weight: 0.8727,
  score: 0.9357,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_022',
  name: 'node_022',
  version: '1.5',
  status: 'stable',
  priority: 9,
  weight: 0.1161,
  score: 0.3413,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_023',
  name: 'node_023',
  version: '3.9',
  status: 'active',
  priority: 10,
  weight: 0.8366,
  score: 0.3843,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_024',
  name: 'node_024',
  version: '2.6',
  status: 'pending',
  priority: 7,
  weight: 0.5138,
  score: 0.2917,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_025',
  name: 'node_025',
  version: '1.4',
  status: 'completed',
  priority: 1,
  weight: 0.1061,
  score: 0.7852,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_026',
  name: 'node_026',
  version: '2.7',
  status: 'recovered',
  priority: 9,
  weight: 0.9599,
  score: 0.7846,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_027',
  name: 'node_027',
  version: '5.4',
  status: 'active',
  priority: 6,
  weight: 0.7258,
  score: 0.1662,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_028',
  name: 'node_028',
  version: '4.7',
  status: 'active',
  priority: 8,
  weight: 0.5783,
  score: 0.5513,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_029',
  name: 'node_029',
  version: '5.0',
  status: 'completed',
  priority: 10,
  weight: 0.8752,
  score: 0.9529,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_030',
  name: 'node_030',
  version: '3.3',
  status: 'degraded',
  priority: 7,
  weight: 0.664,
  score: 0.5764,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_031',
  name: 'node_031',
  version: '3.8',
  status: 'stable',
  priority: 5,
  weight: 0.3451,
  score: 0.2093,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_032',
  name: 'node_032',
  version: '5.4',
  status: 'degraded',
  priority: 10,
  weight: 0.5134,
  score: 0.6055,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_033',
  name: 'node_033',
  version: '4.8',
  status: 'stable',
  priority: 2,
  weight: 0.6702,
  score: 0.5441,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_034',
  name: 'node_034',
  version: '2.5',
  status: 'stable',
  priority: 8,
  weight: 0.2467,
  score: 0.1858,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_035',
  name: 'node_035',
  version: '3.4',
  status: 'degraded',
  priority: 5,
  weight: 0.9252,
  score: 0.0704,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_036',
  name: 'node_036',
  version: '1.7',
  status: 'completed',
  priority: 9,
  weight: 0.1316,
  score: 0.8504,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_037',
  name: 'node_037',
  version: '3.0',
  status: 'stable',
  priority: 7,
  weight: 0.9037,
  score: 0.9542,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_038',
  name: 'node_038',
  version: '4.6',
  status: 'recovered',
  priority: 10,
  weight: 0.6756,
  score: 0.0645,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_07_interface_adapters_1_039',
  name: 'node_039',
  version: '4.5',
  status: 'pending',
  priority: 5,
  weight: 0.108,
  score: 0.3831,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 21,
  createdAt: datetime(),
  active: true
});
