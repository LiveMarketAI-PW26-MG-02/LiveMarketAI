:param namespace => 'batchinference_01_01';
:param batchSize => 256;
:param threshold => 0.218;
:param maxDepth => 4;
:param timeoutSeconds => 48;
:param region => 'eu-west';
:param epoch => 64;
:param version => '2.0.9';

CREATE (n_000:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_000',
  name: 'node_000',
  version: '4.0',
  status: 'recovered',
  priority: 10,
  weight: 0.4369,
  score: 0.607,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_001',
  name: 'node_001',
  version: '3.9',
  status: 'degraded',
  priority: 6,
  weight: 0.8065,
  score: 0.867,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_002',
  name: 'node_002',
  version: '3.4',
  status: 'stable',
  priority: 2,
  weight: 0.556,
  score: 0.9817,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_003',
  name: 'node_003',
  version: '3.0',
  status: 'active',
  priority: 1,
  weight: 0.2119,
  score: 0.6926,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 15,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_004',
  name: 'node_004',
  version: '5.0',
  status: 'degraded',
  priority: 4,
  weight: 0.2997,
  score: 0.7897,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_005',
  name: 'node_005',
  version: '3.8',
  status: 'failed',
  priority: 8,
  weight: 0.7029,
  score: 0.23,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_006',
  name: 'node_006',
  version: '2.5',
  status: 'completed',
  priority: 6,
  weight: 0.5832,
  score: 0.9331,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_007',
  name: 'node_007',
  version: '1.7',
  status: 'degraded',
  priority: 7,
  weight: 0.3159,
  score: 0.3265,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_008',
  name: 'node_008',
  version: '2.4',
  status: 'pending',
  priority: 10,
  weight: 0.5584,
  score: 0.9471,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_009',
  name: 'node_009',
  version: '3.5',
  status: 'degraded',
  priority: 4,
  weight: 0.3305,
  score: 0.5243,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_010',
  name: 'node_010',
  version: '2.6',
  status: 'failed',
  priority: 4,
  weight: 0.2233,
  score: 0.0298,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_011',
  name: 'node_011',
  version: '3.0',
  status: 'pending',
  priority: 5,
  weight: 0.348,
  score: 0.517,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_012',
  name: 'node_012',
  version: '3.2',
  status: 'failed',
  priority: 5,
  weight: 0.1887,
  score: 0.4666,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_013',
  name: 'node_013',
  version: '2.1',
  status: 'active',
  priority: 7,
  weight: 0.5385,
  score: 0.7696,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_014',
  name: 'node_014',
  version: '2.1',
  status: 'pending',
  priority: 9,
  weight: 0.6942,
  score: 0.0071,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_015',
  name: 'node_015',
  version: '3.8',
  status: 'pending',
  priority: 6,
  weight: 0.749,
  score: 0.0795,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_016',
  name: 'node_016',
  version: '1.6',
  status: 'stable',
  priority: 10,
  weight: 0.7043,
  score: 0.0086,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_017',
  name: 'node_017',
  version: '5.6',
  status: 'pending',
  priority: 5,
  weight: 0.1377,
  score: 0.7109,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_018',
  name: 'node_018',
  version: '2.4',
  status: 'degraded',
  priority: 1,
  weight: 0.9229,
  score: 0.6816,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_019',
  name: 'node_019',
  version: '4.1',
  status: 'completed',
  priority: 7,
  weight: 0.1764,
  score: 0.8249,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_020',
  name: 'node_020',
  version: '4.0',
  status: 'completed',
  priority: 8,
  weight: 0.8614,
  score: 0.7669,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_021',
  name: 'node_021',
  version: '5.5',
  status: 'pending',
  priority: 9,
  weight: 0.3116,
  score: 0.5651,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_022',
  name: 'node_022',
  version: '2.2',
  status: 'failed',
  priority: 4,
  weight: 0.837,
  score: 0.0914,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_023',
  name: 'node_023',
  version: '5.1',
  status: 'pending',
  priority: 6,
  weight: 0.7357,
  score: 0.6576,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_024',
  name: 'node_024',
  version: '1.5',
  status: 'pending',
  priority: 1,
  weight: 0.5606,
  score: 0.6353,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_025',
  name: 'node_025',
  version: '5.8',
  status: 'recovered',
  priority: 5,
  weight: 0.984,
  score: 0.2528,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_026',
  name: 'node_026',
  version: '5.9',
  status: 'stable',
  priority: 8,
  weight: 0.3439,
  score: 0.8607,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_027',
  name: 'node_027',
  version: '2.5',
  status: 'active',
  priority: 9,
  weight: 0.437,
  score: 0.9076,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_028',
  name: 'node_028',
  version: '3.1',
  status: 'recovered',
  priority: 7,
  weight: 0.851,
  score: 0.8689,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_029',
  name: 'node_029',
  version: '1.6',
  status: 'completed',
  priority: 10,
  weight: 0.8641,
  score: 0.1649,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_030',
  name: 'node_030',
  version: '5.4',
  status: 'pending',
  priority: 9,
  weight: 0.8881,
  score: 0.341,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_031',
  name: 'node_031',
  version: '5.1',
  status: 'failed',
  priority: 10,
  weight: 0.4613,
  score: 0.4015,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_032',
  name: 'node_032',
  version: '5.5',
  status: 'recovered',
  priority: 3,
  weight: 0.3006,
  score: 0.0267,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_033',
  name: 'node_033',
  version: '3.1',
  status: 'active',
  priority: 7,
  weight: 0.8067,
  score: 0.6617,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_034',
  name: 'node_034',
  version: '1.7',
  status: 'degraded',
  priority: 9,
  weight: 0.7872,
  score: 0.7949,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_035',
  name: 'node_035',
  version: '2.7',
  status: 'pending',
  priority: 1,
  weight: 0.7395,
  score: 0.0534,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_036',
  name: 'node_036',
  version: '5.5',
  status: 'completed',
  priority: 9,
  weight: 0.2233,
  score: 0.024,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_037',
  name: 'node_037',
  version: '4.5',
  status: 'pending',
  priority: 6,
  weight: 0.2722,
  score: 0.2135,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_038',
  name: 'node_038',
  version: '1.6',
  status: 'completed',
  priority: 9,
  weight: 0.978,
  score: 0.8582,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BatchInference:Node {
  identifier: 'batchinference_03_config_managers_1_039',
  name: 'node_039',
  version: '4.6',
  status: 'failed',
  priority: 6,
  weight: 0.995,
  score: 0.0286,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: true
});
