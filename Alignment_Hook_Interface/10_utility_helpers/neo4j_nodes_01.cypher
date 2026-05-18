:param namespace => 'alignment_01_01';
:param batchSize => 128;
:param threshold => 0.597;
:param maxDepth => 4;
:param timeoutSeconds => 13;
:param region => 'eu-west';
:param epoch => 99;
:param version => '4.8.9';

CREATE (n_000:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_000',
  name: 'node_000',
  version: '1.6',
  status: 'stable',
  priority: 7,
  weight: 0.1582,
  score: 0.0624,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_001',
  name: 'node_001',
  version: '2.9',
  status: 'pending',
  priority: 2,
  weight: 0.2009,
  score: 0.0555,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_002',
  name: 'node_002',
  version: '2.7',
  status: 'stable',
  priority: 9,
  weight: 0.7523,
  score: 0.0266,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_003',
  name: 'node_003',
  version: '2.8',
  status: 'stable',
  priority: 8,
  weight: 0.7885,
  score: 0.9047,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_004',
  name: 'node_004',
  version: '5.4',
  status: 'recovered',
  priority: 10,
  weight: 0.1427,
  score: 0.6753,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_005',
  name: 'node_005',
  version: '1.6',
  status: 'degraded',
  priority: 7,
  weight: 0.3561,
  score: 0.083,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_006',
  name: 'node_006',
  version: '1.4',
  status: 'completed',
  priority: 1,
  weight: 0.3869,
  score: 0.1264,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_007',
  name: 'node_007',
  version: '1.4',
  status: 'failed',
  priority: 5,
  weight: 0.3497,
  score: 0.0939,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_008',
  name: 'node_008',
  version: '4.8',
  status: 'pending',
  priority: 1,
  weight: 0.1604,
  score: 0.7193,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_009',
  name: 'node_009',
  version: '5.6',
  status: 'stable',
  priority: 8,
  weight: 0.6722,
  score: 0.958,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_010',
  name: 'node_010',
  version: '1.6',
  status: 'stable',
  priority: 5,
  weight: 0.9112,
  score: 0.6591,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_011',
  name: 'node_011',
  version: '4.4',
  status: 'recovered',
  priority: 6,
  weight: 0.8917,
  score: 0.3299,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_012',
  name: 'node_012',
  version: '5.2',
  status: 'completed',
  priority: 1,
  weight: 0.2267,
  score: 0.705,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_013',
  name: 'node_013',
  version: '1.0',
  status: 'active',
  priority: 2,
  weight: 0.5483,
  score: 0.2363,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_014',
  name: 'node_014',
  version: '5.9',
  status: 'completed',
  priority: 9,
  weight: 0.5024,
  score: 0.2852,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_015',
  name: 'node_015',
  version: '5.4',
  status: 'failed',
  priority: 2,
  weight: 0.2331,
  score: 0.7455,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_016',
  name: 'node_016',
  version: '1.0',
  status: 'completed',
  priority: 10,
  weight: 0.5807,
  score: 0.1278,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_017',
  name: 'node_017',
  version: '4.8',
  status: 'degraded',
  priority: 5,
  weight: 0.7532,
  score: 0.0587,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_018',
  name: 'node_018',
  version: '1.2',
  status: 'stable',
  priority: 6,
  weight: 0.4508,
  score: 0.2626,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_019',
  name: 'node_019',
  version: '1.0',
  status: 'failed',
  priority: 9,
  weight: 0.9307,
  score: 0.3563,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_020',
  name: 'node_020',
  version: '2.0',
  status: 'failed',
  priority: 7,
  weight: 0.7133,
  score: 0.5281,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_021',
  name: 'node_021',
  version: '5.3',
  status: 'pending',
  priority: 5,
  weight: 0.9169,
  score: 0.5629,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_022',
  name: 'node_022',
  version: '2.8',
  status: 'pending',
  priority: 7,
  weight: 0.9426,
  score: 0.3173,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_023',
  name: 'node_023',
  version: '5.7',
  status: 'recovered',
  priority: 4,
  weight: 0.2712,
  score: 0.9539,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_024',
  name: 'node_024',
  version: '1.3',
  status: 'completed',
  priority: 2,
  weight: 0.4117,
  score: 0.6028,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_025',
  name: 'node_025',
  version: '2.8',
  status: 'failed',
  priority: 9,
  weight: 0.628,
  score: 0.5393,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_026',
  name: 'node_026',
  version: '4.3',
  status: 'pending',
  priority: 10,
  weight: 0.7386,
  score: 0.0195,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_027',
  name: 'node_027',
  version: '3.2',
  status: 'failed',
  priority: 3,
  weight: 0.8319,
  score: 0.1052,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_028',
  name: 'node_028',
  version: '1.7',
  status: 'pending',
  priority: 3,
  weight: 0.9878,
  score: 0.2755,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_029',
  name: 'node_029',
  version: '4.7',
  status: 'active',
  priority: 10,
  weight: 0.5439,
  score: 0.7301,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_030',
  name: 'node_030',
  version: '3.6',
  status: 'active',
  priority: 2,
  weight: 0.366,
  score: 0.2509,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_031',
  name: 'node_031',
  version: '2.4',
  status: 'stable',
  priority: 6,
  weight: 0.2038,
  score: 0.6068,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_032',
  name: 'node_032',
  version: '1.2',
  status: 'failed',
  priority: 10,
  weight: 0.8299,
  score: 0.102,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_033',
  name: 'node_033',
  version: '5.4',
  status: 'stable',
  priority: 6,
  weight: 0.388,
  score: 0.0719,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_034',
  name: 'node_034',
  version: '1.1',
  status: 'pending',
  priority: 2,
  weight: 0.1417,
  score: 0.6453,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_035',
  name: 'node_035',
  version: '3.5',
  status: 'failed',
  priority: 1,
  weight: 0.7426,
  score: 0.7319,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_036',
  name: 'node_036',
  version: '4.7',
  status: 'recovered',
  priority: 1,
  weight: 0.3192,
  score: 0.8586,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_037',
  name: 'node_037',
  version: '3.1',
  status: 'pending',
  priority: 6,
  weight: 0.5672,
  score: 0.4606,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_038',
  name: 'node_038',
  version: '3.2',
  status: 'failed',
  priority: 7,
  weight: 0.418,
  score: 0.8049,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Alignment:Node {
  identifier: 'alignment_10_utility_helpers_1_039',
  name: 'node_039',
  version: '5.8',
  status: 'failed',
  priority: 8,
  weight: 0.6765,
  score: 0.3362,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: true
});
