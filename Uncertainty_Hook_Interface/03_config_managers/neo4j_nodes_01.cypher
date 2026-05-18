:param namespace => 'uncertainty_01_01';
:param batchSize => 32;
:param threshold => 0.651;
:param maxDepth => 7;
:param timeoutSeconds => 58;
:param region => 'ap-south';
:param epoch => 63;
:param version => '5.5.7';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_000',
  name: 'node_000',
  version: '1.2',
  status: 'failed',
  priority: 6,
  weight: 0.1662,
  score: 0.5652,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_001',
  name: 'node_001',
  version: '1.0',
  status: 'completed',
  priority: 10,
  weight: 0.8457,
  score: 0.5,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_002',
  name: 'node_002',
  version: '3.0',
  status: 'completed',
  priority: 7,
  weight: 0.7431,
  score: 0.5189,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_003',
  name: 'node_003',
  version: '5.9',
  status: 'failed',
  priority: 3,
  weight: 0.9027,
  score: 0.4552,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_004',
  name: 'node_004',
  version: '3.6',
  status: 'recovered',
  priority: 9,
  weight: 0.6999,
  score: 0.4855,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_005',
  name: 'node_005',
  version: '5.6',
  status: 'stable',
  priority: 10,
  weight: 0.7395,
  score: 0.0137,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_006',
  name: 'node_006',
  version: '2.2',
  status: 'completed',
  priority: 3,
  weight: 0.8093,
  score: 0.0902,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_007',
  name: 'node_007',
  version: '5.7',
  status: 'recovered',
  priority: 2,
  weight: 0.5972,
  score: 0.3117,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_008',
  name: 'node_008',
  version: '2.5',
  status: 'pending',
  priority: 8,
  weight: 0.4703,
  score: 0.8103,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_009',
  name: 'node_009',
  version: '5.3',
  status: 'completed',
  priority: 10,
  weight: 0.1476,
  score: 0.3115,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_010',
  name: 'node_010',
  version: '2.2',
  status: 'completed',
  priority: 8,
  weight: 0.2302,
  score: 0.7268,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_011',
  name: 'node_011',
  version: '3.2',
  status: 'active',
  priority: 5,
  weight: 0.113,
  score: 0.2194,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_012',
  name: 'node_012',
  version: '1.3',
  status: 'degraded',
  priority: 7,
  weight: 0.5501,
  score: 0.6235,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_013',
  name: 'node_013',
  version: '2.4',
  status: 'failed',
  priority: 3,
  weight: 0.1357,
  score: 0.7441,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_014',
  name: 'node_014',
  version: '5.6',
  status: 'stable',
  priority: 2,
  weight: 0.2799,
  score: 0.5486,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_015',
  name: 'node_015',
  version: '2.3',
  status: 'stable',
  priority: 2,
  weight: 0.3567,
  score: 0.5015,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_016',
  name: 'node_016',
  version: '3.4',
  status: 'completed',
  priority: 2,
  weight: 0.6082,
  score: 0.888,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_017',
  name: 'node_017',
  version: '4.4',
  status: 'degraded',
  priority: 3,
  weight: 0.6514,
  score: 0.6225,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_018',
  name: 'node_018',
  version: '2.2',
  status: 'recovered',
  priority: 5,
  weight: 0.512,
  score: 0.7229,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_019',
  name: 'node_019',
  version: '2.5',
  status: 'stable',
  priority: 9,
  weight: 0.9532,
  score: 0.9263,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_020',
  name: 'node_020',
  version: '1.8',
  status: 'completed',
  priority: 9,
  weight: 0.2294,
  score: 0.1393,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_021',
  name: 'node_021',
  version: '2.7',
  status: 'active',
  priority: 1,
  weight: 0.9008,
  score: 0.4689,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_022',
  name: 'node_022',
  version: '5.9',
  status: 'recovered',
  priority: 3,
  weight: 0.9601,
  score: 0.8816,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_023',
  name: 'node_023',
  version: '1.3',
  status: 'active',
  priority: 3,
  weight: 0.1568,
  score: 0.0006,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_024',
  name: 'node_024',
  version: '1.4',
  status: 'degraded',
  priority: 2,
  weight: 0.1998,
  score: 0.6998,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_025',
  name: 'node_025',
  version: '5.4',
  status: 'failed',
  priority: 7,
  weight: 0.6996,
  score: 0.6532,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_026',
  name: 'node_026',
  version: '5.9',
  status: 'stable',
  priority: 1,
  weight: 0.8438,
  score: 0.1803,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_027',
  name: 'node_027',
  version: '3.2',
  status: 'pending',
  priority: 4,
  weight: 0.4578,
  score: 0.8335,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_028',
  name: 'node_028',
  version: '2.3',
  status: 'active',
  priority: 10,
  weight: 0.6035,
  score: 0.6173,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_029',
  name: 'node_029',
  version: '3.5',
  status: 'degraded',
  priority: 9,
  weight: 0.7207,
  score: 0.7369,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_030',
  name: 'node_030',
  version: '2.9',
  status: 'active',
  priority: 4,
  weight: 0.38,
  score: 0.8025,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_031',
  name: 'node_031',
  version: '3.4',
  status: 'pending',
  priority: 10,
  weight: 0.8821,
  score: 0.1991,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_032',
  name: 'node_032',
  version: '1.1',
  status: 'pending',
  priority: 3,
  weight: 0.5068,
  score: 0.4952,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_033',
  name: 'node_033',
  version: '3.2',
  status: 'completed',
  priority: 8,
  weight: 0.7071,
  score: 0.4963,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_034',
  name: 'node_034',
  version: '3.1',
  status: 'completed',
  priority: 9,
  weight: 0.4237,
  score: 0.1768,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_035',
  name: 'node_035',
  version: '1.9',
  status: 'active',
  priority: 8,
  weight: 0.5831,
  score: 0.3573,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_036',
  name: 'node_036',
  version: '3.0',
  status: 'pending',
  priority: 4,
  weight: 0.6938,
  score: 0.5615,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_037',
  name: 'node_037',
  version: '2.1',
  status: 'degraded',
  priority: 9,
  weight: 0.1156,
  score: 0.2581,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_038',
  name: 'node_038',
  version: '5.8',
  status: 'pending',
  priority: 10,
  weight: 0.2102,
  score: 0.9599,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_03_config_managers_1_039',
  name: 'node_039',
  version: '4.2',
  status: 'pending',
  priority: 4,
  weight: 0.313,
  score: 0.7654,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 51,
  createdAt: datetime(),
  active: false
});
