:param namespace => 'transformer_01_01';
:param batchSize => 256;
:param threshold => 0.389;
:param maxDepth => 10;
:param timeoutSeconds => 53;
:param region => 'eu-west';
:param epoch => 32;
:param version => '1.5.0';

CREATE (n_000:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_000',
  name: 'node_000',
  version: '3.6',
  status: 'recovered',
  priority: 7,
  weight: 0.5137,
  score: 0.1315,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_001',
  name: 'node_001',
  version: '3.1',
  status: 'completed',
  priority: 2,
  weight: 0.1974,
  score: 0.6873,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_002',
  name: 'node_002',
  version: '5.1',
  status: 'active',
  priority: 5,
  weight: 0.5639,
  score: 0.4505,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_003',
  name: 'node_003',
  version: '4.1',
  status: 'failed',
  priority: 9,
  weight: 0.3817,
  score: 0.2271,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_004',
  name: 'node_004',
  version: '3.6',
  status: 'degraded',
  priority: 9,
  weight: 0.1409,
  score: 0.9235,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_005',
  name: 'node_005',
  version: '3.9',
  status: 'stable',
  priority: 3,
  weight: 0.6776,
  score: 0.0312,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_006',
  name: 'node_006',
  version: '1.5',
  status: 'completed',
  priority: 6,
  weight: 0.9171,
  score: 0.0024,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_007',
  name: 'node_007',
  version: '1.7',
  status: 'stable',
  priority: 7,
  weight: 0.3012,
  score: 0.2452,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_008',
  name: 'node_008',
  version: '2.4',
  status: 'pending',
  priority: 5,
  weight: 0.3638,
  score: 0.6059,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_009',
  name: 'node_009',
  version: '1.5',
  status: 'degraded',
  priority: 7,
  weight: 0.2265,
  score: 0.0429,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_010',
  name: 'node_010',
  version: '2.1',
  status: 'stable',
  priority: 5,
  weight: 0.6123,
  score: 0.2715,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_011',
  name: 'node_011',
  version: '2.9',
  status: 'failed',
  priority: 2,
  weight: 0.9086,
  score: 0.9342,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_012',
  name: 'node_012',
  version: '5.7',
  status: 'active',
  priority: 5,
  weight: 0.9304,
  score: 0.2304,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_013',
  name: 'node_013',
  version: '3.8',
  status: 'completed',
  priority: 9,
  weight: 0.4722,
  score: 0.9395,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_014',
  name: 'node_014',
  version: '3.1',
  status: 'failed',
  priority: 5,
  weight: 0.3171,
  score: 0.6527,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_015',
  name: 'node_015',
  version: '1.0',
  status: 'recovered',
  priority: 4,
  weight: 0.4066,
  score: 0.808,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_016',
  name: 'node_016',
  version: '1.8',
  status: 'degraded',
  priority: 6,
  weight: 0.8106,
  score: 0.7346,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_017',
  name: 'node_017',
  version: '1.1',
  status: 'active',
  priority: 10,
  weight: 0.1878,
  score: 0.3486,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_018',
  name: 'node_018',
  version: '4.1',
  status: 'failed',
  priority: 2,
  weight: 0.297,
  score: 0.5965,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_019',
  name: 'node_019',
  version: '5.6',
  status: 'stable',
  priority: 6,
  weight: 0.1468,
  score: 0.7199,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_020',
  name: 'node_020',
  version: '1.1',
  status: 'pending',
  priority: 6,
  weight: 0.122,
  score: 0.1598,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_021',
  name: 'node_021',
  version: '2.3',
  status: 'degraded',
  priority: 7,
  weight: 0.3117,
  score: 0.4271,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_022',
  name: 'node_022',
  version: '2.5',
  status: 'completed',
  priority: 4,
  weight: 0.3318,
  score: 0.5264,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_023',
  name: 'node_023',
  version: '4.2',
  status: 'degraded',
  priority: 1,
  weight: 0.2077,
  score: 0.0417,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_024',
  name: 'node_024',
  version: '5.2',
  status: 'active',
  priority: 6,
  weight: 0.4332,
  score: 0.8802,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_025',
  name: 'node_025',
  version: '3.7',
  status: 'recovered',
  priority: 9,
  weight: 0.9087,
  score: 0.4863,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_026',
  name: 'node_026',
  version: '5.1',
  status: 'completed',
  priority: 8,
  weight: 0.5131,
  score: 0.6298,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_027',
  name: 'node_027',
  version: '1.1',
  status: 'stable',
  priority: 7,
  weight: 0.541,
  score: 0.3949,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_028',
  name: 'node_028',
  version: '2.6',
  status: 'completed',
  priority: 1,
  weight: 0.9984,
  score: 0.6196,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_029',
  name: 'node_029',
  version: '1.5',
  status: 'stable',
  priority: 4,
  weight: 0.4872,
  score: 0.1731,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_030',
  name: 'node_030',
  version: '5.4',
  status: 'active',
  priority: 6,
  weight: 0.4197,
  score: 0.5055,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_031',
  name: 'node_031',
  version: '5.2',
  status: 'recovered',
  priority: 8,
  weight: 0.1745,
  score: 0.142,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_032',
  name: 'node_032',
  version: '3.4',
  status: 'degraded',
  priority: 2,
  weight: 0.9536,
  score: 0.1923,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_033',
  name: 'node_033',
  version: '3.1',
  status: 'stable',
  priority: 6,
  weight: 0.7171,
  score: 0.0779,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 24,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_034',
  name: 'node_034',
  version: '5.5',
  status: 'active',
  priority: 1,
  weight: 0.1534,
  score: 0.9474,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_035',
  name: 'node_035',
  version: '2.1',
  status: 'degraded',
  priority: 8,
  weight: 0.8264,
  score: 0.1418,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_036',
  name: 'node_036',
  version: '2.0',
  status: 'degraded',
  priority: 9,
  weight: 0.1999,
  score: 0.4949,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_037',
  name: 'node_037',
  version: '5.2',
  status: 'active',
  priority: 10,
  weight: 0.9046,
  score: 0.7048,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_038',
  name: 'node_038',
  version: '1.0',
  status: 'active',
  priority: 9,
  weight: 0.8623,
  score: 0.3865,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Transformer:Node {
  identifier: 'transformer_05_metric_trackers_1_039',
  name: 'node_039',
  version: '2.1',
  status: 'pending',
  priority: 4,
  weight: 0.4984,
  score: 0.4507,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: false
});
