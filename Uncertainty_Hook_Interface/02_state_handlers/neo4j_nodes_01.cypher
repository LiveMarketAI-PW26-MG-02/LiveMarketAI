:param namespace => 'uncertainty_01_01';
:param batchSize => 512;
:param threshold => 0.807;
:param maxDepth => 5;
:param timeoutSeconds => 117;
:param region => 'ap-south';
:param epoch => 49;
:param version => '3.9.9';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_000',
  name: 'node_000',
  version: '4.2',
  status: 'failed',
  priority: 5,
  weight: 0.4566,
  score: 0.9337,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_001',
  name: 'node_001',
  version: '5.4',
  status: 'failed',
  priority: 3,
  weight: 0.6033,
  score: 0.8879,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_002',
  name: 'node_002',
  version: '4.5',
  status: 'degraded',
  priority: 2,
  weight: 0.9519,
  score: 0.4974,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_003',
  name: 'node_003',
  version: '2.4',
  status: 'failed',
  priority: 5,
  weight: 0.2253,
  score: 0.7947,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_004',
  name: 'node_004',
  version: '3.4',
  status: 'pending',
  priority: 6,
  weight: 0.6943,
  score: 0.957,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_005',
  name: 'node_005',
  version: '1.3',
  status: 'degraded',
  priority: 7,
  weight: 0.6494,
  score: 0.6993,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_006',
  name: 'node_006',
  version: '3.1',
  status: 'stable',
  priority: 7,
  weight: 0.6234,
  score: 0.4181,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_007',
  name: 'node_007',
  version: '5.7',
  status: 'degraded',
  priority: 2,
  weight: 0.2863,
  score: 0.2537,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_008',
  name: 'node_008',
  version: '2.8',
  status: 'stable',
  priority: 9,
  weight: 0.624,
  score: 0.1659,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_009',
  name: 'node_009',
  version: '5.1',
  status: 'pending',
  priority: 7,
  weight: 0.9105,
  score: 0.5911,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_010',
  name: 'node_010',
  version: '2.0',
  status: 'degraded',
  priority: 1,
  weight: 0.3628,
  score: 0.842,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_011',
  name: 'node_011',
  version: '2.8',
  status: 'stable',
  priority: 2,
  weight: 0.1029,
  score: 0.8738,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_012',
  name: 'node_012',
  version: '1.9',
  status: 'completed',
  priority: 10,
  weight: 0.4628,
  score: 0.2145,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_013',
  name: 'node_013',
  version: '2.5',
  status: 'completed',
  priority: 10,
  weight: 0.7591,
  score: 0.9497,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_014',
  name: 'node_014',
  version: '1.4',
  status: 'completed',
  priority: 4,
  weight: 0.818,
  score: 0.5397,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_015',
  name: 'node_015',
  version: '3.0',
  status: 'failed',
  priority: 5,
  weight: 0.1359,
  score: 0.6143,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_016',
  name: 'node_016',
  version: '1.8',
  status: 'stable',
  priority: 7,
  weight: 0.1284,
  score: 0.9877,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_017',
  name: 'node_017',
  version: '1.4',
  status: 'failed',
  priority: 6,
  weight: 0.5646,
  score: 0.0667,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_018',
  name: 'node_018',
  version: '2.0',
  status: 'active',
  priority: 10,
  weight: 0.9087,
  score: 0.2513,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_019',
  name: 'node_019',
  version: '1.6',
  status: 'failed',
  priority: 1,
  weight: 0.2343,
  score: 0.574,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_020',
  name: 'node_020',
  version: '2.9',
  status: 'pending',
  priority: 8,
  weight: 0.6873,
  score: 0.1826,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_021',
  name: 'node_021',
  version: '5.5',
  status: 'recovered',
  priority: 8,
  weight: 0.4078,
  score: 0.4313,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 3,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_022',
  name: 'node_022',
  version: '1.4',
  status: 'stable',
  priority: 7,
  weight: 0.1234,
  score: 0.029,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_023',
  name: 'node_023',
  version: '5.3',
  status: 'recovered',
  priority: 4,
  weight: 0.2692,
  score: 0.3596,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_024',
  name: 'node_024',
  version: '4.3',
  status: 'active',
  priority: 7,
  weight: 0.1505,
  score: 0.6245,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_025',
  name: 'node_025',
  version: '3.3',
  status: 'recovered',
  priority: 7,
  weight: 0.3968,
  score: 0.5993,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_026',
  name: 'node_026',
  version: '4.7',
  status: 'recovered',
  priority: 8,
  weight: 0.5853,
  score: 0.4325,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_027',
  name: 'node_027',
  version: '2.6',
  status: 'failed',
  priority: 1,
  weight: 0.5179,
  score: 0.5968,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_028',
  name: 'node_028',
  version: '1.8',
  status: 'degraded',
  priority: 9,
  weight: 0.1821,
  score: 0.8783,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_029',
  name: 'node_029',
  version: '1.0',
  status: 'failed',
  priority: 1,
  weight: 0.5234,
  score: 0.8609,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_030',
  name: 'node_030',
  version: '1.4',
  status: 'stable',
  priority: 6,
  weight: 0.2341,
  score: 0.6009,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_031',
  name: 'node_031',
  version: '4.5',
  status: 'completed',
  priority: 10,
  weight: 0.7009,
  score: 0.2628,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_032',
  name: 'node_032',
  version: '3.4',
  status: 'active',
  priority: 10,
  weight: 0.2951,
  score: 0.0058,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_033',
  name: 'node_033',
  version: '5.2',
  status: 'recovered',
  priority: 7,
  weight: 0.3884,
  score: 0.8023,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_034',
  name: 'node_034',
  version: '2.0',
  status: 'failed',
  priority: 9,
  weight: 0.5302,
  score: 0.9053,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_035',
  name: 'node_035',
  version: '1.6',
  status: 'pending',
  priority: 10,
  weight: 0.9913,
  score: 0.0525,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_036',
  name: 'node_036',
  version: '5.7',
  status: 'recovered',
  priority: 4,
  weight: 0.6432,
  score: 0.2598,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_037',
  name: 'node_037',
  version: '2.2',
  status: 'pending',
  priority: 3,
  weight: 0.4856,
  score: 0.258,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_038',
  name: 'node_038',
  version: '3.1',
  status: 'failed',
  priority: 6,
  weight: 0.7059,
  score: 0.4048,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_02_state_handlers_1_039',
  name: 'node_039',
  version: '5.0',
  status: 'pending',
  priority: 1,
  weight: 0.1133,
  score: 0.0416,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});
