:param namespace => 'batchinference_01_01';
:param batchSize => 512;
:param threshold => 0.41;
:param maxDepth => 9;
:param timeoutSeconds => 25;
:param region => 'us-west';
:param epoch => 82;
:param version => '1.3.3';

CREATE (n_000:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_000',
  name: 'node_000',
  version: '5.6',
  status: 'pending',
  priority: 3,
  weight: 0.8982,
  score: 0.6843,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_001',
  name: 'node_001',
  version: '2.0',
  status: 'stable',
  priority: 1,
  weight: 0.301,
  score: 0.9137,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_002',
  name: 'node_002',
  version: '3.0',
  status: 'stable',
  priority: 6,
  weight: 0.7857,
  score: 0.878,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_003',
  name: 'node_003',
  version: '5.9',
  status: 'stable',
  priority: 1,
  weight: 0.5672,
  score: 0.7928,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_004',
  name: 'node_004',
  version: '1.7',
  status: 'recovered',
  priority: 4,
  weight: 0.668,
  score: 0.6519,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_005',
  name: 'node_005',
  version: '3.8',
  status: 'recovered',
  priority: 10,
  weight: 0.823,
  score: 0.4036,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_006',
  name: 'node_006',
  version: '3.6',
  status: 'pending',
  priority: 9,
  weight: 0.713,
  score: 0.633,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_007',
  name: 'node_007',
  version: '5.7',
  status: 'stable',
  priority: 7,
  weight: 0.1666,
  score: 0.6709,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_008',
  name: 'node_008',
  version: '4.2',
  status: 'pending',
  priority: 4,
  weight: 0.535,
  score: 0.0527,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_009',
  name: 'node_009',
  version: '4.2',
  status: 'failed',
  priority: 5,
  weight: 0.8486,
  score: 0.7071,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_010',
  name: 'node_010',
  version: '2.3',
  status: 'degraded',
  priority: 1,
  weight: 0.8568,
  score: 0.6664,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_011',
  name: 'node_011',
  version: '3.8',
  status: 'degraded',
  priority: 1,
  weight: 0.1989,
  score: 0.2642,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_012',
  name: 'node_012',
  version: '1.9',
  status: 'degraded',
  priority: 1,
  weight: 0.3748,
  score: 0.4218,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_013',
  name: 'node_013',
  version: '3.9',
  status: 'completed',
  priority: 1,
  weight: 0.2986,
  score: 0.593,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_014',
  name: 'node_014',
  version: '1.6',
  status: 'recovered',
  priority: 2,
  weight: 0.1839,
  score: 0.4573,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_015',
  name: 'node_015',
  version: '4.4',
  status: 'pending',
  priority: 2,
  weight: 0.4576,
  score: 0.7957,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_016',
  name: 'node_016',
  version: '1.0',
  status: 'stable',
  priority: 10,
  weight: 0.5748,
  score: 0.7312,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_017',
  name: 'node_017',
  version: '5.3',
  status: 'degraded',
  priority: 8,
  weight: 0.1769,
  score: 0.6297,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_018',
  name: 'node_018',
  version: '5.0',
  status: 'active',
  priority: 1,
  weight: 0.6995,
  score: 0.1506,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_019',
  name: 'node_019',
  version: '4.7',
  status: 'active',
  priority: 7,
  weight: 0.1951,
  score: 0.8962,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_020',
  name: 'node_020',
  version: '4.6',
  status: 'stable',
  priority: 1,
  weight: 0.3519,
  score: 0.0805,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_021',
  name: 'node_021',
  version: '3.8',
  status: 'active',
  priority: 7,
  weight: 0.6183,
  score: 0.2552,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_022',
  name: 'node_022',
  version: '1.1',
  status: 'stable',
  priority: 4,
  weight: 0.5331,
  score: 0.8633,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_023',
  name: 'node_023',
  version: '5.0',
  status: 'stable',
  priority: 5,
  weight: 0.2871,
  score: 0.0723,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_024',
  name: 'node_024',
  version: '2.9',
  status: 'completed',
  priority: 7,
  weight: 0.6429,
  score: 0.9723,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_025',
  name: 'node_025',
  version: '5.9',
  status: 'pending',
  priority: 2,
  weight: 0.3244,
  score: 0.8095,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_026',
  name: 'node_026',
  version: '4.7',
  status: 'pending',
  priority: 3,
  weight: 0.4595,
  score: 0.8391,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_027',
  name: 'node_027',
  version: '3.0',
  status: 'stable',
  priority: 7,
  weight: 0.8977,
  score: 0.8935,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_028',
  name: 'node_028',
  version: '4.0',
  status: 'recovered',
  priority: 6,
  weight: 0.3041,
  score: 0.9355,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_029',
  name: 'node_029',
  version: '4.3',
  status: 'degraded',
  priority: 5,
  weight: 0.1984,
  score: 0.8728,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_030',
  name: 'node_030',
  version: '2.9',
  status: 'completed',
  priority: 4,
  weight: 0.1526,
  score: 0.4156,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_031',
  name: 'node_031',
  version: '4.0',
  status: 'degraded',
  priority: 10,
  weight: 0.1087,
  score: 0.2681,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_032',
  name: 'node_032',
  version: '1.8',
  status: 'failed',
  priority: 8,
  weight: 0.1207,
  score: 0.2227,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_033',
  name: 'node_033',
  version: '1.4',
  status: 'failed',
  priority: 10,
  weight: 0.5558,
  score: 0.4494,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_034',
  name: 'node_034',
  version: '1.6',
  status: 'degraded',
  priority: 2,
  weight: 0.6639,
  score: 0.0059,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_035',
  name: 'node_035',
  version: '3.7',
  status: 'failed',
  priority: 5,
  weight: 0.3629,
  score: 0.7435,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_036',
  name: 'node_036',
  version: '3.7',
  status: 'degraded',
  priority: 8,
  weight: 0.1847,
  score: 0.4461,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_037',
  name: 'node_037',
  version: '5.3',
  status: 'recovered',
  priority: 1,
  weight: 0.8847,
  score: 0.4493,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_038',
  name: 'node_038',
  version: '2.3',
  status: 'active',
  priority: 1,
  weight: 0.2669,
  score: 0.552,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BatchInference:Node {
  identifier: 'batchinference_05_metric_trackers_1_039',
  name: 'node_039',
  version: '5.9',
  status: 'stable',
  priority: 7,
  weight: 0.7617,
  score: 0.065,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: false
});
