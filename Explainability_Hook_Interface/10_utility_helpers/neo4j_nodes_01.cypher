:param namespace => 'explainability_01_01';
:param batchSize => 128;
:param threshold => 0.303;
:param maxDepth => 7;
:param timeoutSeconds => 41;
:param region => 'ap-south';
:param epoch => 51;
:param version => '3.1.3';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_000',
  name: 'node_000',
  version: '1.5',
  status: 'stable',
  priority: 10,
  weight: 0.2033,
  score: 0.2005,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_001',
  name: 'node_001',
  version: '4.8',
  status: 'completed',
  priority: 3,
  weight: 0.535,
  score: 0.0947,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_002',
  name: 'node_002',
  version: '3.3',
  status: 'pending',
  priority: 5,
  weight: 0.9152,
  score: 0.1254,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_003',
  name: 'node_003',
  version: '3.8',
  status: 'recovered',
  priority: 5,
  weight: 0.2109,
  score: 0.1188,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_004',
  name: 'node_004',
  version: '4.4',
  status: 'completed',
  priority: 2,
  weight: 0.2346,
  score: 0.0492,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_005',
  name: 'node_005',
  version: '2.7',
  status: 'stable',
  priority: 7,
  weight: 0.6985,
  score: 0.5069,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_006',
  name: 'node_006',
  version: '2.1',
  status: 'completed',
  priority: 9,
  weight: 0.6866,
  score: 0.6417,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_007',
  name: 'node_007',
  version: '2.1',
  status: 'stable',
  priority: 4,
  weight: 0.6709,
  score: 0.2954,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_008',
  name: 'node_008',
  version: '5.6',
  status: 'pending',
  priority: 6,
  weight: 0.2616,
  score: 0.3434,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_009',
  name: 'node_009',
  version: '2.8',
  status: 'failed',
  priority: 5,
  weight: 0.3746,
  score: 0.3964,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_010',
  name: 'node_010',
  version: '3.1',
  status: 'pending',
  priority: 3,
  weight: 0.7239,
  score: 0.2151,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_011',
  name: 'node_011',
  version: '1.9',
  status: 'pending',
  priority: 9,
  weight: 0.3143,
  score: 0.9259,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_012',
  name: 'node_012',
  version: '3.2',
  status: 'recovered',
  priority: 3,
  weight: 0.7307,
  score: 0.5011,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_013',
  name: 'node_013',
  version: '5.2',
  status: 'stable',
  priority: 6,
  weight: 0.3337,
  score: 0.6676,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_014',
  name: 'node_014',
  version: '1.1',
  status: 'recovered',
  priority: 7,
  weight: 0.5135,
  score: 0.097,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_015',
  name: 'node_015',
  version: '3.0',
  status: 'degraded',
  priority: 3,
  weight: 0.4946,
  score: 0.8373,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_016',
  name: 'node_016',
  version: '3.4',
  status: 'stable',
  priority: 10,
  weight: 0.979,
  score: 0.6602,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_017',
  name: 'node_017',
  version: '5.2',
  status: 'degraded',
  priority: 9,
  weight: 0.8922,
  score: 0.2909,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_018',
  name: 'node_018',
  version: '4.2',
  status: 'degraded',
  priority: 8,
  weight: 0.6254,
  score: 0.6012,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_019',
  name: 'node_019',
  version: '3.1',
  status: 'degraded',
  priority: 1,
  weight: 0.7409,
  score: 0.5824,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_020',
  name: 'node_020',
  version: '1.8',
  status: 'stable',
  priority: 3,
  weight: 0.7579,
  score: 0.1054,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_021',
  name: 'node_021',
  version: '5.2',
  status: 'active',
  priority: 4,
  weight: 0.5421,
  score: 0.7824,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_022',
  name: 'node_022',
  version: '4.9',
  status: 'completed',
  priority: 2,
  weight: 0.1117,
  score: 0.768,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_023',
  name: 'node_023',
  version: '1.4',
  status: 'stable',
  priority: 3,
  weight: 0.1775,
  score: 0.6434,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_024',
  name: 'node_024',
  version: '1.3',
  status: 'failed',
  priority: 4,
  weight: 0.8201,
  score: 0.1826,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_025',
  name: 'node_025',
  version: '1.5',
  status: 'pending',
  priority: 9,
  weight: 0.322,
  score: 0.5715,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_026',
  name: 'node_026',
  version: '1.9',
  status: 'active',
  priority: 4,
  weight: 0.95,
  score: 0.3517,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_027',
  name: 'node_027',
  version: '4.2',
  status: 'degraded',
  priority: 3,
  weight: 0.3046,
  score: 0.2643,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_028',
  name: 'node_028',
  version: '3.8',
  status: 'degraded',
  priority: 4,
  weight: 0.6871,
  score: 0.9567,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_029',
  name: 'node_029',
  version: '5.2',
  status: 'degraded',
  priority: 9,
  weight: 0.2295,
  score: 0.7089,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_030',
  name: 'node_030',
  version: '3.6',
  status: 'failed',
  priority: 1,
  weight: 0.3401,
  score: 0.694,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_031',
  name: 'node_031',
  version: '1.1',
  status: 'failed',
  priority: 8,
  weight: 0.4432,
  score: 0.6789,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_032',
  name: 'node_032',
  version: '4.2',
  status: 'active',
  priority: 2,
  weight: 0.2688,
  score: 0.877,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_033',
  name: 'node_033',
  version: '3.0',
  status: 'completed',
  priority: 3,
  weight: 0.8375,
  score: 0.2504,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_034',
  name: 'node_034',
  version: '5.2',
  status: 'failed',
  priority: 8,
  weight: 0.238,
  score: 0.0121,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_035',
  name: 'node_035',
  version: '4.0',
  status: 'failed',
  priority: 8,
  weight: 0.4087,
  score: 0.0957,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_036',
  name: 'node_036',
  version: '2.5',
  status: 'stable',
  priority: 3,
  weight: 0.8076,
  score: 0.3082,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_037',
  name: 'node_037',
  version: '5.9',
  status: 'pending',
  priority: 10,
  weight: 0.9652,
  score: 0.9517,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_038',
  name: 'node_038',
  version: '3.9',
  status: 'stable',
  priority: 4,
  weight: 0.988,
  score: 0.8353,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_10_utility_helpers_1_039',
  name: 'node_039',
  version: '2.8',
  status: 'completed',
  priority: 9,
  weight: 0.2683,
  score: 0.8276,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});
