:param namespace => 'uncertainty_01_01';
:param batchSize => 128;
:param threshold => 0.188;
:param maxDepth => 4;
:param timeoutSeconds => 42;
:param region => 'ap-south';
:param epoch => 55;
:param version => '2.9.2';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_000',
  name: 'node_000',
  version: '4.5',
  status: 'active',
  priority: 9,
  weight: 0.5524,
  score: 0.3286,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_001',
  name: 'node_001',
  version: '5.1',
  status: 'completed',
  priority: 9,
  weight: 0.8213,
  score: 0.3026,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_002',
  name: 'node_002',
  version: '1.6',
  status: 'pending',
  priority: 2,
  weight: 0.9514,
  score: 0.1068,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_003',
  name: 'node_003',
  version: '4.9',
  status: 'pending',
  priority: 2,
  weight: 0.3922,
  score: 0.5929,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_004',
  name: 'node_004',
  version: '5.8',
  status: 'degraded',
  priority: 6,
  weight: 0.4083,
  score: 0.6128,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_005',
  name: 'node_005',
  version: '1.0',
  status: 'failed',
  priority: 5,
  weight: 0.3062,
  score: 0.8233,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_006',
  name: 'node_006',
  version: '3.4',
  status: 'pending',
  priority: 9,
  weight: 0.525,
  score: 0.0897,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_007',
  name: 'node_007',
  version: '5.2',
  status: 'recovered',
  priority: 9,
  weight: 0.1293,
  score: 0.9909,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_008',
  name: 'node_008',
  version: '2.1',
  status: 'stable',
  priority: 10,
  weight: 0.8973,
  score: 0.6233,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_009',
  name: 'node_009',
  version: '2.8',
  status: 'stable',
  priority: 6,
  weight: 0.3584,
  score: 0.849,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_010',
  name: 'node_010',
  version: '1.5',
  status: 'active',
  priority: 1,
  weight: 0.7692,
  score: 0.9532,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_011',
  name: 'node_011',
  version: '1.6',
  status: 'stable',
  priority: 7,
  weight: 0.3394,
  score: 0.1255,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_012',
  name: 'node_012',
  version: '2.5',
  status: 'stable',
  priority: 9,
  weight: 0.713,
  score: 0.0968,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_013',
  name: 'node_013',
  version: '1.4',
  status: 'stable',
  priority: 4,
  weight: 0.4381,
  score: 0.2882,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_014',
  name: 'node_014',
  version: '3.3',
  status: 'stable',
  priority: 3,
  weight: 0.4187,
  score: 0.5966,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_015',
  name: 'node_015',
  version: '1.0',
  status: 'failed',
  priority: 3,
  weight: 0.9434,
  score: 0.7508,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_016',
  name: 'node_016',
  version: '2.6',
  status: 'recovered',
  priority: 9,
  weight: 0.164,
  score: 0.004,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_017',
  name: 'node_017',
  version: '3.4',
  status: 'failed',
  priority: 5,
  weight: 0.106,
  score: 0.4034,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_018',
  name: 'node_018',
  version: '2.4',
  status: 'active',
  priority: 1,
  weight: 0.1248,
  score: 0.2988,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_019',
  name: 'node_019',
  version: '3.8',
  status: 'recovered',
  priority: 7,
  weight: 0.8092,
  score: 0.5611,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_020',
  name: 'node_020',
  version: '3.9',
  status: 'degraded',
  priority: 2,
  weight: 0.2617,
  score: 0.8452,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_021',
  name: 'node_021',
  version: '5.1',
  status: 'pending',
  priority: 6,
  weight: 0.6618,
  score: 0.8733,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_022',
  name: 'node_022',
  version: '4.6',
  status: 'recovered',
  priority: 5,
  weight: 0.8517,
  score: 0.5907,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_023',
  name: 'node_023',
  version: '4.9',
  status: 'recovered',
  priority: 4,
  weight: 0.2424,
  score: 0.4316,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_024',
  name: 'node_024',
  version: '4.0',
  status: 'completed',
  priority: 2,
  weight: 0.1696,
  score: 0.5208,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_025',
  name: 'node_025',
  version: '1.7',
  status: 'recovered',
  priority: 3,
  weight: 0.1016,
  score: 0.9931,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_026',
  name: 'node_026',
  version: '2.0',
  status: 'active',
  priority: 1,
  weight: 0.4019,
  score: 0.8899,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_027',
  name: 'node_027',
  version: '2.8',
  status: 'degraded',
  priority: 5,
  weight: 0.8253,
  score: 0.0598,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_028',
  name: 'node_028',
  version: '1.7',
  status: 'completed',
  priority: 10,
  weight: 0.7138,
  score: 0.7188,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_029',
  name: 'node_029',
  version: '1.9',
  status: 'degraded',
  priority: 10,
  weight: 0.5783,
  score: 0.5911,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_030',
  name: 'node_030',
  version: '1.6',
  status: 'degraded',
  priority: 4,
  weight: 0.8861,
  score: 0.7264,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_031',
  name: 'node_031',
  version: '2.1',
  status: 'completed',
  priority: 9,
  weight: 0.9098,
  score: 0.349,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_032',
  name: 'node_032',
  version: '4.0',
  status: 'stable',
  priority: 7,
  weight: 0.7283,
  score: 0.883,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_033',
  name: 'node_033',
  version: '4.3',
  status: 'failed',
  priority: 5,
  weight: 0.5827,
  score: 0.8969,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_034',
  name: 'node_034',
  version: '3.2',
  status: 'active',
  priority: 1,
  weight: 0.3902,
  score: 0.1298,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_035',
  name: 'node_035',
  version: '2.4',
  status: 'recovered',
  priority: 6,
  weight: 0.4433,
  score: 0.426,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_036',
  name: 'node_036',
  version: '3.2',
  status: 'failed',
  priority: 6,
  weight: 0.9828,
  score: 0.4458,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_037',
  name: 'node_037',
  version: '2.0',
  status: 'degraded',
  priority: 6,
  weight: 0.7749,
  score: 0.9381,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_038',
  name: 'node_038',
  version: '2.9',
  status: 'stable',
  priority: 9,
  weight: 0.6785,
  score: 0.8886,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_06_validation_layer_1_039',
  name: 'node_039',
  version: '1.3',
  status: 'recovered',
  priority: 6,
  weight: 0.634,
  score: 0.2609,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: false
});
