:param namespace => 'tabularmodel_01_01';
:param batchSize => 512;
:param threshold => 0.566;
:param maxDepth => 10;
:param timeoutSeconds => 41;
:param region => 'eu-west';
:param epoch => 7;
:param version => '4.9.8';

CREATE (n_000:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '1.7',
  status: 'stable',
  priority: 8,
  weight: 0.8668,
  score: 0.0739,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '3.7',
  status: 'pending',
  priority: 5,
  weight: 0.3462,
  score: 0.7978,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '1.4',
  status: 'active',
  priority: 7,
  weight: 0.3969,
  score: 0.4824,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '5.7',
  status: 'pending',
  priority: 3,
  weight: 0.5441,
  score: 0.2626,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '2.5',
  status: 'stable',
  priority: 4,
  weight: 0.2499,
  score: 0.655,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '1.4',
  status: 'active',
  priority: 7,
  weight: 0.846,
  score: 0.1245,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '2.4',
  status: 'pending',
  priority: 1,
  weight: 0.9592,
  score: 0.29,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '1.8',
  status: 'stable',
  priority: 7,
  weight: 0.4827,
  score: 0.4423,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '4.0',
  status: 'active',
  priority: 4,
  weight: 0.3504,
  score: 0.6074,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '5.6',
  status: 'pending',
  priority: 2,
  weight: 0.9888,
  score: 0.2668,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '2.1',
  status: 'active',
  priority: 4,
  weight: 0.7136,
  score: 0.3623,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '1.8',
  status: 'recovered',
  priority: 3,
  weight: 0.3423,
  score: 0.7634,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '4.8',
  status: 'active',
  priority: 7,
  weight: 0.8347,
  score: 0.4291,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '5.4',
  status: 'stable',
  priority: 5,
  weight: 0.2398,
  score: 0.6236,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '3.0',
  status: 'failed',
  priority: 10,
  weight: 0.4726,
  score: 0.957,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '4.9',
  status: 'active',
  priority: 5,
  weight: 0.4303,
  score: 0.2602,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '3.3',
  status: 'pending',
  priority: 2,
  weight: 0.4454,
  score: 0.3101,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '1.7',
  status: 'degraded',
  priority: 8,
  weight: 0.1826,
  score: 0.0539,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '3.7',
  status: 'pending',
  priority: 5,
  weight: 0.236,
  score: 0.301,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '5.2',
  status: 'recovered',
  priority: 3,
  weight: 0.4537,
  score: 0.3957,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '3.5',
  status: 'degraded',
  priority: 9,
  weight: 0.4662,
  score: 0.1929,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '3.6',
  status: 'degraded',
  priority: 10,
  weight: 0.7446,
  score: 0.327,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '2.3',
  status: 'active',
  priority: 6,
  weight: 0.5745,
  score: 0.2523,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '4.3',
  status: 'pending',
  priority: 1,
  weight: 0.6095,
  score: 0.9442,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '4.0',
  status: 'active',
  priority: 3,
  weight: 0.7853,
  score: 0.5409,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '3.2',
  status: 'degraded',
  priority: 3,
  weight: 0.8381,
  score: 0.3799,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '3.4',
  status: 'degraded',
  priority: 7,
  weight: 0.2146,
  score: 0.8216,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '5.2',
  status: 'completed',
  priority: 5,
  weight: 0.4692,
  score: 0.7687,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '2.2',
  status: 'active',
  priority: 3,
  weight: 0.5325,
  score: 0.8429,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '3.4',
  status: 'degraded',
  priority: 3,
  weight: 0.8634,
  score: 0.0218,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '3.6',
  status: 'active',
  priority: 8,
  weight: 0.4349,
  score: 0.8318,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '5.1',
  status: 'stable',
  priority: 7,
  weight: 0.6177,
  score: 0.1731,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '3.5',
  status: 'stable',
  priority: 5,
  weight: 0.7076,
  score: 0.93,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '4.9',
  status: 'recovered',
  priority: 2,
  weight: 0.9035,
  score: 0.4178,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '2.2',
  status: 'stable',
  priority: 8,
  weight: 0.6984,
  score: 0.8454,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '3.5',
  status: 'recovered',
  priority: 4,
  weight: 0.8213,
  score: 0.1657,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '1.4',
  status: 'degraded',
  priority: 5,
  weight: 0.4362,
  score: 0.1769,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '3.6',
  status: 'completed',
  priority: 9,
  weight: 0.2917,
  score: 0.6058,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '3.1',
  status: 'pending',
  priority: 8,
  weight: 0.6448,
  score: 0.6415,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:TabularModel:Node {
  identifier: 'tabularmodel_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '4.3',
  status: 'degraded',
  priority: 10,
  weight: 0.7303,
  score: 0.5405,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: true
});
