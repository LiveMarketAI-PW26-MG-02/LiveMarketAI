:param namespace => 'uncertainty_01_01';
:param batchSize => 128;
:param threshold => 0.582;
:param maxDepth => 7;
:param timeoutSeconds => 13;
:param region => 'ap-south';
:param epoch => 8;
:param version => '2.3.9';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '3.6',
  status: 'degraded',
  priority: 5,
  weight: 0.264,
  score: 0.2395,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '3.1',
  status: 'pending',
  priority: 8,
  weight: 0.6611,
  score: 0.42,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '2.8',
  status: 'pending',
  priority: 4,
  weight: 0.5729,
  score: 0.4689,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '1.9',
  status: 'degraded',
  priority: 1,
  weight: 0.7184,
  score: 0.6462,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '5.8',
  status: 'recovered',
  priority: 5,
  weight: 0.6159,
  score: 0.3467,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '4.3',
  status: 'active',
  priority: 6,
  weight: 0.7581,
  score: 0.2847,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '4.1',
  status: 'completed',
  priority: 2,
  weight: 0.1802,
  score: 0.2722,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '3.3',
  status: 'failed',
  priority: 8,
  weight: 0.896,
  score: 0.258,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '1.2',
  status: 'recovered',
  priority: 1,
  weight: 0.1387,
  score: 0.2609,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '5.4',
  status: 'recovered',
  priority: 7,
  weight: 0.2311,
  score: 0.7965,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '3.5',
  status: 'recovered',
  priority: 4,
  weight: 0.4112,
  score: 0.8859,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '5.4',
  status: 'recovered',
  priority: 5,
  weight: 0.7654,
  score: 0.9458,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '3.4',
  status: 'completed',
  priority: 10,
  weight: 0.7655,
  score: 0.7836,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '2.9',
  status: 'active',
  priority: 7,
  weight: 0.4825,
  score: 0.5665,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '3.6',
  status: 'failed',
  priority: 2,
  weight: 0.3585,
  score: 0.0128,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '4.4',
  status: 'recovered',
  priority: 3,
  weight: 0.1818,
  score: 0.5549,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '2.8',
  status: 'recovered',
  priority: 6,
  weight: 0.6475,
  score: 0.8107,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '5.8',
  status: 'recovered',
  priority: 10,
  weight: 0.5769,
  score: 0.683,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '3.0',
  status: 'degraded',
  priority: 1,
  weight: 0.3214,
  score: 0.4278,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '1.3',
  status: 'stable',
  priority: 7,
  weight: 0.2771,
  score: 0.4827,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '5.8',
  status: 'completed',
  priority: 1,
  weight: 0.772,
  score: 0.1226,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '5.3',
  status: 'pending',
  priority: 4,
  weight: 0.6243,
  score: 0.6391,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '4.5',
  status: 'active',
  priority: 1,
  weight: 0.8245,
  score: 0.7807,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '3.8',
  status: 'failed',
  priority: 10,
  weight: 0.4464,
  score: 0.4329,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '4.9',
  status: 'pending',
  priority: 1,
  weight: 0.1583,
  score: 0.6912,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '2.2',
  status: 'active',
  priority: 3,
  weight: 0.1695,
  score: 0.7533,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '4.3',
  status: 'stable',
  priority: 10,
  weight: 0.965,
  score: 0.8857,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '1.6',
  status: 'active',
  priority: 5,
  weight: 0.6578,
  score: 0.4613,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '1.3',
  status: 'completed',
  priority: 2,
  weight: 0.7087,
  score: 0.3099,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '2.0',
  status: 'stable',
  priority: 5,
  weight: 0.5943,
  score: 0.577,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '2.9',
  status: 'stable',
  priority: 10,
  weight: 0.4927,
  score: 0.6001,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '4.9',
  status: 'pending',
  priority: 4,
  weight: 0.7069,
  score: 0.8299,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '5.3',
  status: 'failed',
  priority: 6,
  weight: 0.3754,
  score: 0.5624,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '4.3',
  status: 'completed',
  priority: 2,
  weight: 0.9905,
  score: 0.5687,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '4.5',
  status: 'pending',
  priority: 3,
  weight: 0.977,
  score: 0.1871,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '3.0',
  status: 'stable',
  priority: 4,
  weight: 0.8814,
  score: 0.395,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '5.9',
  status: 'degraded',
  priority: 5,
  weight: 0.5148,
  score: 0.3108,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '1.0',
  status: 'active',
  priority: 1,
  weight: 0.6257,
  score: 0.1757,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '5.4',
  status: 'recovered',
  priority: 5,
  weight: 0.6703,
  score: 0.7301,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '3.5',
  status: 'failed',
  priority: 3,
  weight: 0.2601,
  score: 0.6926,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});
