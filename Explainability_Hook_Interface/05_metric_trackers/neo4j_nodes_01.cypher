:param namespace => 'explainability_01_01';
:param batchSize => 512;
:param threshold => 0.805;
:param maxDepth => 7;
:param timeoutSeconds => 16;
:param region => 'ap-south';
:param epoch => 10;
:param version => '1.5.0';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_000',
  name: 'node_000',
  version: '5.9',
  status: 'degraded',
  priority: 9,
  weight: 0.111,
  score: 0.4171,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_001',
  name: 'node_001',
  version: '3.3',
  status: 'pending',
  priority: 10,
  weight: 0.63,
  score: 0.3694,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_002',
  name: 'node_002',
  version: '3.0',
  status: 'stable',
  priority: 9,
  weight: 0.5696,
  score: 0.1042,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_003',
  name: 'node_003',
  version: '3.0',
  status: 'stable',
  priority: 9,
  weight: 0.4828,
  score: 0.3562,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_004',
  name: 'node_004',
  version: '5.0',
  status: 'completed',
  priority: 2,
  weight: 0.1724,
  score: 0.0933,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_005',
  name: 'node_005',
  version: '5.1',
  status: 'failed',
  priority: 4,
  weight: 0.107,
  score: 0.7167,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_006',
  name: 'node_006',
  version: '3.3',
  status: 'stable',
  priority: 8,
  weight: 0.5789,
  score: 0.8054,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_007',
  name: 'node_007',
  version: '1.0',
  status: 'stable',
  priority: 6,
  weight: 0.6404,
  score: 0.3899,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_008',
  name: 'node_008',
  version: '3.1',
  status: 'degraded',
  priority: 2,
  weight: 0.7969,
  score: 0.3218,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_009',
  name: 'node_009',
  version: '1.1',
  status: 'failed',
  priority: 9,
  weight: 0.5925,
  score: 0.683,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_010',
  name: 'node_010',
  version: '5.6',
  status: 'degraded',
  priority: 7,
  weight: 0.8265,
  score: 0.6816,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_011',
  name: 'node_011',
  version: '3.0',
  status: 'failed',
  priority: 9,
  weight: 0.3117,
  score: 0.3044,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_012',
  name: 'node_012',
  version: '4.3',
  status: 'active',
  priority: 3,
  weight: 0.339,
  score: 0.7663,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_013',
  name: 'node_013',
  version: '4.4',
  status: 'recovered',
  priority: 10,
  weight: 0.7298,
  score: 0.4496,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_014',
  name: 'node_014',
  version: '5.2',
  status: 'pending',
  priority: 2,
  weight: 0.4707,
  score: 0.0736,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_015',
  name: 'node_015',
  version: '5.7',
  status: 'active',
  priority: 1,
  weight: 0.2262,
  score: 0.3777,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_016',
  name: 'node_016',
  version: '1.8',
  status: 'recovered',
  priority: 9,
  weight: 0.7234,
  score: 0.9577,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_017',
  name: 'node_017',
  version: '1.6',
  status: 'stable',
  priority: 2,
  weight: 0.7647,
  score: 0.0082,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_018',
  name: 'node_018',
  version: '3.7',
  status: 'recovered',
  priority: 9,
  weight: 0.2492,
  score: 0.7489,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_019',
  name: 'node_019',
  version: '1.6',
  status: 'active',
  priority: 4,
  weight: 0.4053,
  score: 0.0374,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_020',
  name: 'node_020',
  version: '4.6',
  status: 'recovered',
  priority: 5,
  weight: 0.2754,
  score: 0.9544,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_021',
  name: 'node_021',
  version: '3.6',
  status: 'degraded',
  priority: 5,
  weight: 0.2837,
  score: 0.3756,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_022',
  name: 'node_022',
  version: '5.2',
  status: 'pending',
  priority: 1,
  weight: 0.8836,
  score: 0.2118,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_023',
  name: 'node_023',
  version: '3.4',
  status: 'completed',
  priority: 8,
  weight: 0.3652,
  score: 0.1613,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_024',
  name: 'node_024',
  version: '1.6',
  status: 'degraded',
  priority: 3,
  weight: 0.9213,
  score: 0.4724,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_025',
  name: 'node_025',
  version: '4.8',
  status: 'stable',
  priority: 5,
  weight: 0.7162,
  score: 0.4696,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_026',
  name: 'node_026',
  version: '3.2',
  status: 'failed',
  priority: 1,
  weight: 0.1224,
  score: 0.4388,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_027',
  name: 'node_027',
  version: '3.4',
  status: 'recovered',
  priority: 2,
  weight: 0.3087,
  score: 0.1135,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_028',
  name: 'node_028',
  version: '1.9',
  status: 'active',
  priority: 10,
  weight: 0.356,
  score: 0.0502,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_029',
  name: 'node_029',
  version: '4.3',
  status: 'recovered',
  priority: 5,
  weight: 0.3714,
  score: 0.467,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_030',
  name: 'node_030',
  version: '1.0',
  status: 'active',
  priority: 1,
  weight: 0.9009,
  score: 0.6123,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_031',
  name: 'node_031',
  version: '1.1',
  status: 'pending',
  priority: 4,
  weight: 0.292,
  score: 0.7165,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_032',
  name: 'node_032',
  version: '2.5',
  status: 'stable',
  priority: 7,
  weight: 0.9502,
  score: 0.7698,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_033',
  name: 'node_033',
  version: '1.4',
  status: 'failed',
  priority: 9,
  weight: 0.1025,
  score: 0.3263,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_034',
  name: 'node_034',
  version: '4.4',
  status: 'degraded',
  priority: 3,
  weight: 0.5172,
  score: 0.0271,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_035',
  name: 'node_035',
  version: '3.0',
  status: 'recovered',
  priority: 5,
  weight: 0.5639,
  score: 0.1181,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_036',
  name: 'node_036',
  version: '1.8',
  status: 'stable',
  priority: 6,
  weight: 0.7655,
  score: 0.6143,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_037',
  name: 'node_037',
  version: '3.3',
  status: 'completed',
  priority: 2,
  weight: 0.8708,
  score: 0.2154,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_038',
  name: 'node_038',
  version: '5.6',
  status: 'recovered',
  priority: 6,
  weight: 0.2151,
  score: 0.2899,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_05_metric_trackers_1_039',
  name: 'node_039',
  version: '1.4',
  status: 'stable',
  priority: 5,
  weight: 0.2328,
  score: 0.3906,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: true
});
