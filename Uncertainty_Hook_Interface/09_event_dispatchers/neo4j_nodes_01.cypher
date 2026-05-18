:param namespace => 'uncertainty_01_01';
:param batchSize => 64;
:param threshold => 0.579;
:param maxDepth => 12;
:param timeoutSeconds => 119;
:param region => 'ap-south';
:param epoch => 31;
:param version => '5.3.4';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_000',
  name: 'node_000',
  version: '4.2',
  status: 'active',
  priority: 7,
  weight: 0.7973,
  score: 0.689,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_001',
  name: 'node_001',
  version: '1.9',
  status: 'stable',
  priority: 1,
  weight: 0.4479,
  score: 0.7077,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_002',
  name: 'node_002',
  version: '2.5',
  status: 'completed',
  priority: 10,
  weight: 0.6492,
  score: 0.9785,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_003',
  name: 'node_003',
  version: '1.7',
  status: 'active',
  priority: 2,
  weight: 0.9563,
  score: 0.2708,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_004',
  name: 'node_004',
  version: '2.2',
  status: 'degraded',
  priority: 8,
  weight: 0.8485,
  score: 0.8499,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_005',
  name: 'node_005',
  version: '4.0',
  status: 'pending',
  priority: 9,
  weight: 0.8404,
  score: 0.5327,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_006',
  name: 'node_006',
  version: '5.3',
  status: 'pending',
  priority: 5,
  weight: 0.4411,
  score: 0.9236,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_007',
  name: 'node_007',
  version: '3.0',
  status: 'stable',
  priority: 7,
  weight: 0.7236,
  score: 0.7151,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_008',
  name: 'node_008',
  version: '3.7',
  status: 'active',
  priority: 1,
  weight: 0.4191,
  score: 0.5134,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_009',
  name: 'node_009',
  version: '5.7',
  status: 'recovered',
  priority: 4,
  weight: 0.4398,
  score: 0.3564,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_010',
  name: 'node_010',
  version: '2.0',
  status: 'active',
  priority: 7,
  weight: 0.5542,
  score: 0.1167,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_011',
  name: 'node_011',
  version: '1.8',
  status: 'pending',
  priority: 7,
  weight: 0.5143,
  score: 0.3911,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_012',
  name: 'node_012',
  version: '4.1',
  status: 'completed',
  priority: 6,
  weight: 0.4702,
  score: 0.5097,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_013',
  name: 'node_013',
  version: '5.0',
  status: 'degraded',
  priority: 3,
  weight: 0.8475,
  score: 0.5863,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_014',
  name: 'node_014',
  version: '5.5',
  status: 'completed',
  priority: 5,
  weight: 0.8999,
  score: 0.8299,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_015',
  name: 'node_015',
  version: '3.3',
  status: 'active',
  priority: 5,
  weight: 0.956,
  score: 0.1935,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_016',
  name: 'node_016',
  version: '4.0',
  status: 'stable',
  priority: 4,
  weight: 0.571,
  score: 0.5514,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_017',
  name: 'node_017',
  version: '3.3',
  status: 'recovered',
  priority: 4,
  weight: 0.6158,
  score: 0.3125,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_018',
  name: 'node_018',
  version: '4.3',
  status: 'degraded',
  priority: 10,
  weight: 0.6831,
  score: 0.657,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_019',
  name: 'node_019',
  version: '5.2',
  status: 'recovered',
  priority: 2,
  weight: 0.2206,
  score: 0.9159,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_020',
  name: 'node_020',
  version: '4.8',
  status: 'degraded',
  priority: 6,
  weight: 0.6689,
  score: 0.3409,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_021',
  name: 'node_021',
  version: '3.6',
  status: 'completed',
  priority: 1,
  weight: 0.9974,
  score: 0.9724,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_022',
  name: 'node_022',
  version: '5.3',
  status: 'pending',
  priority: 3,
  weight: 0.1588,
  score: 0.1637,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_023',
  name: 'node_023',
  version: '5.8',
  status: 'failed',
  priority: 2,
  weight: 0.4771,
  score: 0.5157,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_024',
  name: 'node_024',
  version: '2.4',
  status: 'stable',
  priority: 2,
  weight: 0.76,
  score: 0.8059,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_025',
  name: 'node_025',
  version: '1.4',
  status: 'stable',
  priority: 10,
  weight: 0.1898,
  score: 0.2721,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_026',
  name: 'node_026',
  version: '3.7',
  status: 'active',
  priority: 7,
  weight: 0.8479,
  score: 0.4885,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_027',
  name: 'node_027',
  version: '4.7',
  status: 'pending',
  priority: 3,
  weight: 0.1402,
  score: 0.5809,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_028',
  name: 'node_028',
  version: '3.2',
  status: 'failed',
  priority: 5,
  weight: 0.8295,
  score: 0.9847,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_029',
  name: 'node_029',
  version: '4.7',
  status: 'completed',
  priority: 6,
  weight: 0.7211,
  score: 0.9441,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_030',
  name: 'node_030',
  version: '1.9',
  status: 'failed',
  priority: 2,
  weight: 0.4818,
  score: 0.5183,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_031',
  name: 'node_031',
  version: '2.2',
  status: 'failed',
  priority: 9,
  weight: 0.878,
  score: 0.5159,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_032',
  name: 'node_032',
  version: '4.1',
  status: 'recovered',
  priority: 9,
  weight: 0.1725,
  score: 0.294,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_033',
  name: 'node_033',
  version: '4.5',
  status: 'degraded',
  priority: 4,
  weight: 0.5414,
  score: 0.4904,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_034',
  name: 'node_034',
  version: '4.7',
  status: 'completed',
  priority: 4,
  weight: 0.3085,
  score: 0.9348,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_035',
  name: 'node_035',
  version: '4.0',
  status: 'recovered',
  priority: 10,
  weight: 0.1154,
  score: 0.2005,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_036',
  name: 'node_036',
  version: '1.9',
  status: 'completed',
  priority: 2,
  weight: 0.6419,
  score: 0.8866,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_037',
  name: 'node_037',
  version: '3.3',
  status: 'active',
  priority: 9,
  weight: 0.1511,
  score: 0.41,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_038',
  name: 'node_038',
  version: '3.6',
  status: 'recovered',
  priority: 4,
  weight: 0.3192,
  score: 0.7315,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_09_event_dispatchers_1_039',
  name: 'node_039',
  version: '4.7',
  status: 'completed',
  priority: 1,
  weight: 0.9094,
  score: 0.9653,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: false
});
