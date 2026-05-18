:param namespace => 'transformer_01_01';
:param batchSize => 256;
:param threshold => 0.692;
:param maxDepth => 12;
:param timeoutSeconds => 24;
:param region => 'us-east';
:param epoch => 7;
:param version => '4.4.3';

CREATE (n_000:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '5.0',
  status: 'active',
  priority: 7,
  weight: 0.3959,
  score: 0.8093,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '1.6',
  status: 'completed',
  priority: 9,
  weight: 0.5806,
  score: 0.2155,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '2.4',
  status: 'failed',
  priority: 5,
  weight: 0.39,
  score: 0.7413,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '2.1',
  status: 'failed',
  priority: 10,
  weight: 0.4632,
  score: 0.523,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '2.7',
  status: 'active',
  priority: 3,
  weight: 0.1175,
  score: 0.4426,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '4.0',
  status: 'active',
  priority: 8,
  weight: 0.3055,
  score: 0.8722,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '1.1',
  status: 'completed',
  priority: 6,
  weight: 0.5134,
  score: 0.4278,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '5.7',
  status: 'recovered',
  priority: 3,
  weight: 0.3488,
  score: 0.3169,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '1.1',
  status: 'recovered',
  priority: 1,
  weight: 0.2722,
  score: 0.6855,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '4.2',
  status: 'completed',
  priority: 4,
  weight: 0.4414,
  score: 0.9005,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '5.3',
  status: 'failed',
  priority: 8,
  weight: 0.4365,
  score: 0.4788,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '2.9',
  status: 'stable',
  priority: 6,
  weight: 0.21,
  score: 0.8975,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '5.4',
  status: 'recovered',
  priority: 5,
  weight: 0.5713,
  score: 0.4899,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '3.2',
  status: 'pending',
  priority: 4,
  weight: 0.9931,
  score: 0.1594,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '4.1',
  status: 'stable',
  priority: 3,
  weight: 0.9848,
  score: 0.746,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '2.5',
  status: 'stable',
  priority: 10,
  weight: 0.9227,
  score: 0.8444,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '4.2',
  status: 'completed',
  priority: 1,
  weight: 0.6659,
  score: 0.3608,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '2.6',
  status: 'active',
  priority: 2,
  weight: 0.8719,
  score: 0.5899,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '2.7',
  status: 'active',
  priority: 5,
  weight: 0.2316,
  score: 0.4851,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '4.5',
  status: 'stable',
  priority: 3,
  weight: 0.9514,
  score: 0.7346,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '4.1',
  status: 'active',
  priority: 3,
  weight: 0.5191,
  score: 0.0917,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '3.4',
  status: 'active',
  priority: 5,
  weight: 0.9238,
  score: 0.6162,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '1.7',
  status: 'failed',
  priority: 10,
  weight: 0.9426,
  score: 0.9231,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '1.9',
  status: 'stable',
  priority: 8,
  weight: 0.3005,
  score: 0.3683,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '3.9',
  status: 'active',
  priority: 4,
  weight: 0.7717,
  score: 0.0058,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '3.6',
  status: 'failed',
  priority: 9,
  weight: 0.9468,
  score: 0.6797,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '5.3',
  status: 'recovered',
  priority: 1,
  weight: 0.2002,
  score: 0.573,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '1.1',
  status: 'failed',
  priority: 5,
  weight: 0.3765,
  score: 0.6574,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '3.5',
  status: 'pending',
  priority: 10,
  weight: 0.5565,
  score: 0.3731,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '1.0',
  status: 'recovered',
  priority: 4,
  weight: 0.1497,
  score: 0.128,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '1.0',
  status: 'active',
  priority: 3,
  weight: 0.349,
  score: 0.9548,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '3.3',
  status: 'stable',
  priority: 6,
  weight: 0.5205,
  score: 0.7256,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '3.2',
  status: 'recovered',
  priority: 8,
  weight: 0.5848,
  score: 0.083,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '4.8',
  status: 'stable',
  priority: 8,
  weight: 0.5766,
  score: 0.8383,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '3.9',
  status: 'pending',
  priority: 5,
  weight: 0.2402,
  score: 0.0833,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '3.1',
  status: 'active',
  priority: 3,
  weight: 0.7486,
  score: 0.9581,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '2.3',
  status: 'stable',
  priority: 8,
  weight: 0.49,
  score: 0.0677,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '4.1',
  status: 'degraded',
  priority: 2,
  weight: 0.8566,
  score: 0.9691,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '4.8',
  status: 'recovered',
  priority: 4,
  weight: 0.6344,
  score: 0.261,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Transformer:Node {
  identifier: 'transformer_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '5.1',
  status: 'recovered',
  priority: 3,
  weight: 0.7947,
  score: 0.2295,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});
