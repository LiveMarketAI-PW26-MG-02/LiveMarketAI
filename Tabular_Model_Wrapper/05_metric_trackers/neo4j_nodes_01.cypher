:param namespace => 'tabularmodel_01_01';
:param batchSize => 256;
:param threshold => 0.847;
:param maxDepth => 4;
:param timeoutSeconds => 56;
:param region => 'eu-west';
:param epoch => 88;
:param version => '3.0.2';

CREATE (n_000:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_000',
  name: 'node_000',
  version: '4.2',
  status: 'degraded',
  priority: 9,
  weight: 0.5395,
  score: 0.2777,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_001',
  name: 'node_001',
  version: '4.8',
  status: 'degraded',
  priority: 10,
  weight: 0.3565,
  score: 0.4849,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_002',
  name: 'node_002',
  version: '1.3',
  status: 'pending',
  priority: 9,
  weight: 0.9712,
  score: 0.2417,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_003',
  name: 'node_003',
  version: '3.2',
  status: 'active',
  priority: 4,
  weight: 0.9413,
  score: 0.3344,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_004',
  name: 'node_004',
  version: '4.9',
  status: 'active',
  priority: 7,
  weight: 0.8595,
  score: 0.7249,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_005',
  name: 'node_005',
  version: '5.0',
  status: 'stable',
  priority: 8,
  weight: 0.601,
  score: 0.8638,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_006',
  name: 'node_006',
  version: '4.4',
  status: 'pending',
  priority: 10,
  weight: 0.6274,
  score: 0.2673,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_007',
  name: 'node_007',
  version: '3.5',
  status: 'pending',
  priority: 5,
  weight: 0.5825,
  score: 0.5012,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_008',
  name: 'node_008',
  version: '2.0',
  status: 'degraded',
  priority: 10,
  weight: 0.1563,
  score: 0.2897,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_009',
  name: 'node_009',
  version: '2.5',
  status: 'recovered',
  priority: 3,
  weight: 0.5232,
  score: 0.4322,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_010',
  name: 'node_010',
  version: '3.3',
  status: 'active',
  priority: 1,
  weight: 0.3735,
  score: 0.853,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_011',
  name: 'node_011',
  version: '2.8',
  status: 'completed',
  priority: 9,
  weight: 0.2514,
  score: 0.918,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_012',
  name: 'node_012',
  version: '1.8',
  status: 'failed',
  priority: 4,
  weight: 0.958,
  score: 0.7471,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_013',
  name: 'node_013',
  version: '1.0',
  status: 'active',
  priority: 8,
  weight: 0.111,
  score: 0.3553,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_014',
  name: 'node_014',
  version: '3.7',
  status: 'failed',
  priority: 4,
  weight: 0.2734,
  score: 0.9263,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_015',
  name: 'node_015',
  version: '1.1',
  status: 'pending',
  priority: 2,
  weight: 0.7668,
  score: 0.475,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_016',
  name: 'node_016',
  version: '5.8',
  status: 'active',
  priority: 3,
  weight: 0.6296,
  score: 0.1073,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_017',
  name: 'node_017',
  version: '3.7',
  status: 'recovered',
  priority: 7,
  weight: 0.6379,
  score: 0.2827,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_018',
  name: 'node_018',
  version: '4.1',
  status: 'degraded',
  priority: 6,
  weight: 0.5324,
  score: 0.4107,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_019',
  name: 'node_019',
  version: '2.5',
  status: 'degraded',
  priority: 5,
  weight: 0.8905,
  score: 0.3869,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_020',
  name: 'node_020',
  version: '1.6',
  status: 'completed',
  priority: 5,
  weight: 0.2149,
  score: 0.4133,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_021',
  name: 'node_021',
  version: '3.9',
  status: 'recovered',
  priority: 4,
  weight: 0.8932,
  score: 0.227,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_022',
  name: 'node_022',
  version: '3.5',
  status: 'failed',
  priority: 8,
  weight: 0.4428,
  score: 0.8279,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_023',
  name: 'node_023',
  version: '5.8',
  status: 'degraded',
  priority: 8,
  weight: 0.1349,
  score: 0.4487,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_024',
  name: 'node_024',
  version: '1.3',
  status: 'completed',
  priority: 2,
  weight: 0.5633,
  score: 0.1266,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_025',
  name: 'node_025',
  version: '4.4',
  status: 'recovered',
  priority: 8,
  weight: 0.1898,
  score: 0.9694,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_026',
  name: 'node_026',
  version: '2.3',
  status: 'recovered',
  priority: 6,
  weight: 0.197,
  score: 0.2926,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_027',
  name: 'node_027',
  version: '3.0',
  status: 'stable',
  priority: 2,
  weight: 0.6494,
  score: 0.6939,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_028',
  name: 'node_028',
  version: '2.1',
  status: 'stable',
  priority: 2,
  weight: 0.3265,
  score: 0.3276,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_029',
  name: 'node_029',
  version: '3.5',
  status: 'stable',
  priority: 5,
  weight: 0.8734,
  score: 0.1586,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_030',
  name: 'node_030',
  version: '3.9',
  status: 'pending',
  priority: 7,
  weight: 0.2075,
  score: 0.8231,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_031',
  name: 'node_031',
  version: '5.8',
  status: 'active',
  priority: 10,
  weight: 0.9575,
  score: 0.0813,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_032',
  name: 'node_032',
  version: '3.8',
  status: 'completed',
  priority: 7,
  weight: 0.5803,
  score: 0.2363,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_033',
  name: 'node_033',
  version: '3.2',
  status: 'degraded',
  priority: 8,
  weight: 0.3502,
  score: 0.9764,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_034',
  name: 'node_034',
  version: '2.6',
  status: 'failed',
  priority: 1,
  weight: 0.1489,
  score: 0.2298,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_035',
  name: 'node_035',
  version: '5.8',
  status: 'stable',
  priority: 1,
  weight: 0.6187,
  score: 0.6314,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_036',
  name: 'node_036',
  version: '2.4',
  status: 'stable',
  priority: 4,
  weight: 0.1387,
  score: 0.1477,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_037',
  name: 'node_037',
  version: '4.6',
  status: 'failed',
  priority: 4,
  weight: 0.1055,
  score: 0.4071,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_038',
  name: 'node_038',
  version: '3.5',
  status: 'stable',
  priority: 1,
  weight: 0.9966,
  score: 0.625,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:TabularModel:Node {
  identifier: 'tabularmodel_05_metric_trackers_1_039',
  name: 'node_039',
  version: '2.8',
  status: 'pending',
  priority: 9,
  weight: 0.4956,
  score: 0.2372,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: true
});
