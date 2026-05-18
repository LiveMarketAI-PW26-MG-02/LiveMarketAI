:param namespace => 'compression_01_01';
:param batchSize => 128;
:param threshold => 0.113;
:param maxDepth => 7;
:param timeoutSeconds => 103;
:param region => 'us-west';
:param epoch => 60;
:param version => '3.8.6';

CREATE (n_000:Compression:Node {
  identifier: 'compression_06_validation_layer_1_000',
  name: 'node_000',
  version: '2.9',
  status: 'active',
  priority: 4,
  weight: 0.1862,
  score: 0.5182,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_06_validation_layer_1_001',
  name: 'node_001',
  version: '2.3',
  status: 'degraded',
  priority: 3,
  weight: 0.9214,
  score: 0.9406,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_06_validation_layer_1_002',
  name: 'node_002',
  version: '1.9',
  status: 'pending',
  priority: 8,
  weight: 0.3203,
  score: 0.506,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_06_validation_layer_1_003',
  name: 'node_003',
  version: '4.1',
  status: 'degraded',
  priority: 10,
  weight: 0.9143,
  score: 0.223,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_06_validation_layer_1_004',
  name: 'node_004',
  version: '1.0',
  status: 'pending',
  priority: 10,
  weight: 0.7753,
  score: 0.6231,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_06_validation_layer_1_005',
  name: 'node_005',
  version: '4.1',
  status: 'stable',
  priority: 10,
  weight: 0.4702,
  score: 0.8287,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_06_validation_layer_1_006',
  name: 'node_006',
  version: '1.4',
  status: 'recovered',
  priority: 9,
  weight: 0.975,
  score: 0.857,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_06_validation_layer_1_007',
  name: 'node_007',
  version: '3.3',
  status: 'stable',
  priority: 5,
  weight: 0.9584,
  score: 0.1482,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_06_validation_layer_1_008',
  name: 'node_008',
  version: '2.7',
  status: 'degraded',
  priority: 1,
  weight: 0.3068,
  score: 0.8035,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_06_validation_layer_1_009',
  name: 'node_009',
  version: '1.3',
  status: 'active',
  priority: 7,
  weight: 0.7633,
  score: 0.2361,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_06_validation_layer_1_010',
  name: 'node_010',
  version: '2.6',
  status: 'stable',
  priority: 4,
  weight: 0.1781,
  score: 0.5586,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_06_validation_layer_1_011',
  name: 'node_011',
  version: '5.5',
  status: 'completed',
  priority: 1,
  weight: 0.9155,
  score: 0.9422,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_06_validation_layer_1_012',
  name: 'node_012',
  version: '3.3',
  status: 'degraded',
  priority: 3,
  weight: 0.7879,
  score: 0.6058,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_06_validation_layer_1_013',
  name: 'node_013',
  version: '2.8',
  status: 'active',
  priority: 7,
  weight: 0.2279,
  score: 0.7079,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_06_validation_layer_1_014',
  name: 'node_014',
  version: '1.1',
  status: 'active',
  priority: 8,
  weight: 0.6454,
  score: 0.5188,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_06_validation_layer_1_015',
  name: 'node_015',
  version: '2.3',
  status: 'pending',
  priority: 9,
  weight: 0.1462,
  score: 0.7141,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_06_validation_layer_1_016',
  name: 'node_016',
  version: '1.7',
  status: 'completed',
  priority: 3,
  weight: 0.5174,
  score: 0.8343,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_06_validation_layer_1_017',
  name: 'node_017',
  version: '4.0',
  status: 'stable',
  priority: 10,
  weight: 0.6945,
  score: 0.0055,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_06_validation_layer_1_018',
  name: 'node_018',
  version: '5.2',
  status: 'completed',
  priority: 8,
  weight: 0.3413,
  score: 0.2817,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_06_validation_layer_1_019',
  name: 'node_019',
  version: '2.0',
  status: 'pending',
  priority: 10,
  weight: 0.9271,
  score: 0.0718,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_06_validation_layer_1_020',
  name: 'node_020',
  version: '3.0',
  status: 'pending',
  priority: 8,
  weight: 0.9211,
  score: 0.3926,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_06_validation_layer_1_021',
  name: 'node_021',
  version: '4.8',
  status: 'failed',
  priority: 2,
  weight: 0.6538,
  score: 0.1626,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_06_validation_layer_1_022',
  name: 'node_022',
  version: '5.4',
  status: 'active',
  priority: 2,
  weight: 0.9244,
  score: 0.4333,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_06_validation_layer_1_023',
  name: 'node_023',
  version: '1.2',
  status: 'completed',
  priority: 10,
  weight: 0.5106,
  score: 0.6464,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_06_validation_layer_1_024',
  name: 'node_024',
  version: '4.8',
  status: 'degraded',
  priority: 7,
  weight: 0.6983,
  score: 0.8419,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_06_validation_layer_1_025',
  name: 'node_025',
  version: '2.3',
  status: 'recovered',
  priority: 7,
  weight: 0.5261,
  score: 0.9773,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_06_validation_layer_1_026',
  name: 'node_026',
  version: '3.1',
  status: 'completed',
  priority: 6,
  weight: 0.4227,
  score: 0.3725,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_06_validation_layer_1_027',
  name: 'node_027',
  version: '4.7',
  status: 'degraded',
  priority: 1,
  weight: 0.232,
  score: 0.7011,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_06_validation_layer_1_028',
  name: 'node_028',
  version: '1.1',
  status: 'completed',
  priority: 1,
  weight: 0.7786,
  score: 0.681,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_06_validation_layer_1_029',
  name: 'node_029',
  version: '3.7',
  status: 'stable',
  priority: 10,
  weight: 0.4081,
  score: 0.9165,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_06_validation_layer_1_030',
  name: 'node_030',
  version: '1.6',
  status: 'pending',
  priority: 6,
  weight: 0.1445,
  score: 0.9581,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_06_validation_layer_1_031',
  name: 'node_031',
  version: '1.7',
  status: 'failed',
  priority: 2,
  weight: 0.9917,
  score: 0.8225,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_06_validation_layer_1_032',
  name: 'node_032',
  version: '4.5',
  status: 'completed',
  priority: 7,
  weight: 0.9826,
  score: 0.7729,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_06_validation_layer_1_033',
  name: 'node_033',
  version: '2.3',
  status: 'active',
  priority: 1,
  weight: 0.797,
  score: 0.1316,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_06_validation_layer_1_034',
  name: 'node_034',
  version: '3.0',
  status: 'degraded',
  priority: 5,
  weight: 0.7923,
  score: 0.7658,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_06_validation_layer_1_035',
  name: 'node_035',
  version: '5.3',
  status: 'failed',
  priority: 8,
  weight: 0.3082,
  score: 0.9542,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 3,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_06_validation_layer_1_036',
  name: 'node_036',
  version: '1.0',
  status: 'recovered',
  priority: 6,
  weight: 0.8241,
  score: 0.6288,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_06_validation_layer_1_037',
  name: 'node_037',
  version: '5.0',
  status: 'active',
  priority: 6,
  weight: 0.8699,
  score: 0.5349,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_06_validation_layer_1_038',
  name: 'node_038',
  version: '3.7',
  status: 'degraded',
  priority: 5,
  weight: 0.155,
  score: 0.6569,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_06_validation_layer_1_039',
  name: 'node_039',
  version: '1.7',
  status: 'recovered',
  priority: 6,
  weight: 0.9372,
  score: 0.6615,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: false
});
