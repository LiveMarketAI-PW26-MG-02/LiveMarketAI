:param namespace => 'uncertainty_01_01';
:param batchSize => 256;
:param threshold => 0.159;
:param maxDepth => 5;
:param timeoutSeconds => 43;
:param region => 'us-west';
:param epoch => 64;
:param version => '3.9.0';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_000',
  name: 'node_000',
  version: '3.6',
  status: 'stable',
  priority: 5,
  weight: 0.7767,
  score: 0.9657,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_001',
  name: 'node_001',
  version: '3.8',
  status: 'active',
  priority: 7,
  weight: 0.4668,
  score: 0.8513,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_002',
  name: 'node_002',
  version: '2.2',
  status: 'failed',
  priority: 1,
  weight: 0.5803,
  score: 0.1795,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_003',
  name: 'node_003',
  version: '4.3',
  status: 'degraded',
  priority: 3,
  weight: 0.3306,
  score: 0.6319,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_004',
  name: 'node_004',
  version: '4.7',
  status: 'stable',
  priority: 4,
  weight: 0.8149,
  score: 0.624,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_005',
  name: 'node_005',
  version: '4.0',
  status: 'recovered',
  priority: 3,
  weight: 0.6777,
  score: 0.0986,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_006',
  name: 'node_006',
  version: '2.7',
  status: 'completed',
  priority: 1,
  weight: 0.7757,
  score: 0.1069,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_007',
  name: 'node_007',
  version: '4.9',
  status: 'stable',
  priority: 10,
  weight: 0.5607,
  score: 0.9578,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_008',
  name: 'node_008',
  version: '1.7',
  status: 'degraded',
  priority: 6,
  weight: 0.3213,
  score: 0.2705,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_009',
  name: 'node_009',
  version: '5.7',
  status: 'degraded',
  priority: 4,
  weight: 0.3625,
  score: 0.488,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_010',
  name: 'node_010',
  version: '2.9',
  status: 'active',
  priority: 3,
  weight: 0.1321,
  score: 0.4173,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_011',
  name: 'node_011',
  version: '4.5',
  status: 'active',
  priority: 2,
  weight: 0.6704,
  score: 0.8414,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_012',
  name: 'node_012',
  version: '4.1',
  status: 'degraded',
  priority: 4,
  weight: 0.2572,
  score: 0.79,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_013',
  name: 'node_013',
  version: '3.8',
  status: 'completed',
  priority: 6,
  weight: 0.7471,
  score: 0.0106,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_014',
  name: 'node_014',
  version: '3.8',
  status: 'pending',
  priority: 8,
  weight: 0.3298,
  score: 0.6773,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_015',
  name: 'node_015',
  version: '3.7',
  status: 'pending',
  priority: 8,
  weight: 0.9724,
  score: 0.8798,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_016',
  name: 'node_016',
  version: '3.6',
  status: 'pending',
  priority: 1,
  weight: 0.2592,
  score: 0.1888,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_017',
  name: 'node_017',
  version: '1.7',
  status: 'stable',
  priority: 4,
  weight: 0.4573,
  score: 0.5969,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_018',
  name: 'node_018',
  version: '3.5',
  status: 'failed',
  priority: 3,
  weight: 0.4861,
  score: 0.4149,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_019',
  name: 'node_019',
  version: '1.7',
  status: 'active',
  priority: 9,
  weight: 0.1615,
  score: 0.6367,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_020',
  name: 'node_020',
  version: '4.0',
  status: 'completed',
  priority: 6,
  weight: 0.5207,
  score: 0.4737,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_021',
  name: 'node_021',
  version: '3.1',
  status: 'stable',
  priority: 10,
  weight: 0.4733,
  score: 0.2083,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_022',
  name: 'node_022',
  version: '3.2',
  status: 'pending',
  priority: 7,
  weight: 0.1688,
  score: 0.6999,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_023',
  name: 'node_023',
  version: '2.5',
  status: 'completed',
  priority: 4,
  weight: 0.9278,
  score: 0.9279,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_024',
  name: 'node_024',
  version: '5.2',
  status: 'pending',
  priority: 9,
  weight: 0.7822,
  score: 0.7639,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_025',
  name: 'node_025',
  version: '4.4',
  status: 'recovered',
  priority: 5,
  weight: 0.764,
  score: 0.118,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_026',
  name: 'node_026',
  version: '2.7',
  status: 'recovered',
  priority: 5,
  weight: 0.4497,
  score: 0.6105,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_027',
  name: 'node_027',
  version: '4.6',
  status: 'stable',
  priority: 5,
  weight: 0.894,
  score: 0.6056,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_028',
  name: 'node_028',
  version: '2.5',
  status: 'pending',
  priority: 2,
  weight: 0.713,
  score: 0.0409,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_029',
  name: 'node_029',
  version: '2.2',
  status: 'active',
  priority: 1,
  weight: 0.3956,
  score: 0.0832,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_030',
  name: 'node_030',
  version: '5.0',
  status: 'active',
  priority: 9,
  weight: 0.5286,
  score: 0.2862,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_031',
  name: 'node_031',
  version: '5.9',
  status: 'recovered',
  priority: 2,
  weight: 0.2649,
  score: 0.0336,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_032',
  name: 'node_032',
  version: '4.6',
  status: 'failed',
  priority: 8,
  weight: 0.4695,
  score: 0.6088,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_033',
  name: 'node_033',
  version: '3.4',
  status: 'degraded',
  priority: 3,
  weight: 0.8359,
  score: 0.2026,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_034',
  name: 'node_034',
  version: '3.9',
  status: 'pending',
  priority: 6,
  weight: 0.9473,
  score: 0.3119,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_035',
  name: 'node_035',
  version: '5.2',
  status: 'recovered',
  priority: 8,
  weight: 0.2628,
  score: 0.9964,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_036',
  name: 'node_036',
  version: '4.1',
  status: 'recovered',
  priority: 9,
  weight: 0.695,
  score: 0.2786,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_037',
  name: 'node_037',
  version: '5.2',
  status: 'completed',
  priority: 10,
  weight: 0.8787,
  score: 0.1625,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_038',
  name: 'node_038',
  version: '5.2',
  status: 'degraded',
  priority: 6,
  weight: 0.4163,
  score: 0.3734,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_05_metric_trackers_1_039',
  name: 'node_039',
  version: '2.1',
  status: 'failed',
  priority: 5,
  weight: 0.6839,
  score: 0.1977,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 16,
  createdAt: datetime(),
  active: true
});
