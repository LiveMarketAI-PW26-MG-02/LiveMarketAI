:param namespace => 'explainability_01_01';
:param batchSize => 64;
:param threshold => 0.183;
:param maxDepth => 7;
:param timeoutSeconds => 118;
:param region => 'us-west';
:param epoch => 30;
:param version => '4.5.1';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_000',
  name: 'node_000',
  version: '4.4',
  status: 'stable',
  priority: 5,
  weight: 0.5244,
  score: 0.4086,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_001',
  name: 'node_001',
  version: '1.0',
  status: 'completed',
  priority: 1,
  weight: 0.7205,
  score: 0.4718,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_002',
  name: 'node_002',
  version: '5.9',
  status: 'recovered',
  priority: 5,
  weight: 0.1375,
  score: 0.4499,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_003',
  name: 'node_003',
  version: '1.4',
  status: 'stable',
  priority: 10,
  weight: 0.1939,
  score: 0.3801,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_004',
  name: 'node_004',
  version: '1.6',
  status: 'failed',
  priority: 4,
  weight: 0.6945,
  score: 0.7412,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_005',
  name: 'node_005',
  version: '3.4',
  status: 'degraded',
  priority: 10,
  weight: 0.6779,
  score: 0.771,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_006',
  name: 'node_006',
  version: '5.4',
  status: 'completed',
  priority: 6,
  weight: 0.1764,
  score: 0.8665,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_007',
  name: 'node_007',
  version: '3.5',
  status: 'stable',
  priority: 1,
  weight: 0.9824,
  score: 0.0476,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_008',
  name: 'node_008',
  version: '3.9',
  status: 'pending',
  priority: 7,
  weight: 0.1758,
  score: 0.9745,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_009',
  name: 'node_009',
  version: '2.4',
  status: 'failed',
  priority: 1,
  weight: 0.6621,
  score: 0.36,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_010',
  name: 'node_010',
  version: '3.8',
  status: 'recovered',
  priority: 9,
  weight: 0.835,
  score: 0.7097,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_011',
  name: 'node_011',
  version: '4.9',
  status: 'stable',
  priority: 1,
  weight: 0.1664,
  score: 0.5218,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_012',
  name: 'node_012',
  version: '2.2',
  status: 'degraded',
  priority: 3,
  weight: 0.9004,
  score: 0.7565,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_013',
  name: 'node_013',
  version: '5.7',
  status: 'failed',
  priority: 3,
  weight: 0.8406,
  score: 0.013,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_014',
  name: 'node_014',
  version: '5.3',
  status: 'stable',
  priority: 1,
  weight: 0.6224,
  score: 0.0668,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_015',
  name: 'node_015',
  version: '2.4',
  status: 'completed',
  priority: 10,
  weight: 0.927,
  score: 0.4902,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_016',
  name: 'node_016',
  version: '4.5',
  status: 'recovered',
  priority: 7,
  weight: 0.6445,
  score: 0.559,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_017',
  name: 'node_017',
  version: '1.6',
  status: 'degraded',
  priority: 7,
  weight: 0.6643,
  score: 0.3973,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_018',
  name: 'node_018',
  version: '2.8',
  status: 'active',
  priority: 4,
  weight: 0.2031,
  score: 0.2947,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_019',
  name: 'node_019',
  version: '2.6',
  status: 'pending',
  priority: 3,
  weight: 0.1141,
  score: 0.2955,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_020',
  name: 'node_020',
  version: '1.8',
  status: 'degraded',
  priority: 3,
  weight: 0.4009,
  score: 0.4651,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_021',
  name: 'node_021',
  version: '5.2',
  status: 'stable',
  priority: 10,
  weight: 0.6999,
  score: 0.1554,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_022',
  name: 'node_022',
  version: '4.4',
  status: 'degraded',
  priority: 4,
  weight: 0.6704,
  score: 0.4697,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_023',
  name: 'node_023',
  version: '2.5',
  status: 'stable',
  priority: 5,
  weight: 0.7813,
  score: 0.2491,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_024',
  name: 'node_024',
  version: '4.1',
  status: 'failed',
  priority: 6,
  weight: 0.537,
  score: 0.1012,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_025',
  name: 'node_025',
  version: '4.3',
  status: 'degraded',
  priority: 7,
  weight: 0.9238,
  score: 0.8064,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_026',
  name: 'node_026',
  version: '1.0',
  status: 'pending',
  priority: 9,
  weight: 0.165,
  score: 0.6796,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_027',
  name: 'node_027',
  version: '3.6',
  status: 'pending',
  priority: 3,
  weight: 0.5103,
  score: 0.4343,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_028',
  name: 'node_028',
  version: '1.1',
  status: 'pending',
  priority: 4,
  weight: 0.4823,
  score: 0.4626,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_029',
  name: 'node_029',
  version: '3.5',
  status: 'failed',
  priority: 7,
  weight: 0.4612,
  score: 0.6367,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_030',
  name: 'node_030',
  version: '2.5',
  status: 'recovered',
  priority: 10,
  weight: 0.2203,
  score: 0.8475,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_031',
  name: 'node_031',
  version: '5.5',
  status: 'degraded',
  priority: 6,
  weight: 0.5797,
  score: 0.6501,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_032',
  name: 'node_032',
  version: '4.0',
  status: 'failed',
  priority: 7,
  weight: 0.1628,
  score: 0.2528,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_033',
  name: 'node_033',
  version: '1.6',
  status: 'recovered',
  priority: 3,
  weight: 0.1944,
  score: 0.1463,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_034',
  name: 'node_034',
  version: '3.5',
  status: 'completed',
  priority: 1,
  weight: 0.7855,
  score: 0.0878,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_035',
  name: 'node_035',
  version: '4.2',
  status: 'completed',
  priority: 6,
  weight: 0.2689,
  score: 0.7946,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_036',
  name: 'node_036',
  version: '5.3',
  status: 'failed',
  priority: 3,
  weight: 0.4692,
  score: 0.7671,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_037',
  name: 'node_037',
  version: '2.4',
  status: 'degraded',
  priority: 8,
  weight: 0.8553,
  score: 0.4952,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_038',
  name: 'node_038',
  version: '5.8',
  status: 'recovered',
  priority: 2,
  weight: 0.5567,
  score: 0.9189,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_06_validation_layer_1_039',
  name: 'node_039',
  version: '4.2',
  status: 'recovered',
  priority: 7,
  weight: 0.1223,
  score: 0.1743,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 51,
  createdAt: datetime(),
  active: false
});
