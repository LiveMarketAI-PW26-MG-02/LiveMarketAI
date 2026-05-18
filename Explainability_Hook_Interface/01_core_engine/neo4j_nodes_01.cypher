:param namespace => 'explainability_01_01';
:param batchSize => 512;
:param threshold => 0.238;
:param maxDepth => 10;
:param timeoutSeconds => 83;
:param region => 'us-west';
:param epoch => 86;
:param version => '4.4.7';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_000',
  name: 'node_000',
  version: '2.2',
  status: 'active',
  priority: 4,
  weight: 0.7715,
  score: 0.9864,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_001',
  name: 'node_001',
  version: '3.3',
  status: 'active',
  priority: 10,
  weight: 0.5409,
  score: 0.7647,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_002',
  name: 'node_002',
  version: '4.4',
  status: 'failed',
  priority: 9,
  weight: 0.2177,
  score: 0.0333,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_003',
  name: 'node_003',
  version: '1.5',
  status: 'active',
  priority: 7,
  weight: 0.1072,
  score: 0.4555,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_004',
  name: 'node_004',
  version: '3.8',
  status: 'active',
  priority: 4,
  weight: 0.4979,
  score: 0.7784,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_005',
  name: 'node_005',
  version: '3.1',
  status: 'degraded',
  priority: 5,
  weight: 0.7109,
  score: 0.1722,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_006',
  name: 'node_006',
  version: '1.0',
  status: 'active',
  priority: 9,
  weight: 0.6295,
  score: 0.1862,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_007',
  name: 'node_007',
  version: '3.7',
  status: 'completed',
  priority: 2,
  weight: 0.5836,
  score: 0.5171,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_008',
  name: 'node_008',
  version: '4.2',
  status: 'recovered',
  priority: 9,
  weight: 0.4279,
  score: 0.1177,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_009',
  name: 'node_009',
  version: '4.1',
  status: 'stable',
  priority: 10,
  weight: 0.414,
  score: 0.5914,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_010',
  name: 'node_010',
  version: '3.2',
  status: 'active',
  priority: 7,
  weight: 0.9694,
  score: 0.6611,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_011',
  name: 'node_011',
  version: '5.1',
  status: 'completed',
  priority: 2,
  weight: 0.2923,
  score: 0.5825,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_012',
  name: 'node_012',
  version: '1.0',
  status: 'completed',
  priority: 2,
  weight: 0.7591,
  score: 0.6647,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_013',
  name: 'node_013',
  version: '1.9',
  status: 'active',
  priority: 5,
  weight: 0.7474,
  score: 0.8985,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_014',
  name: 'node_014',
  version: '2.5',
  status: 'failed',
  priority: 8,
  weight: 0.9293,
  score: 0.8984,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_015',
  name: 'node_015',
  version: '4.4',
  status: 'stable',
  priority: 3,
  weight: 0.3019,
  score: 0.7937,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_016',
  name: 'node_016',
  version: '1.9',
  status: 'stable',
  priority: 5,
  weight: 0.1843,
  score: 0.307,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_017',
  name: 'node_017',
  version: '2.2',
  status: 'stable',
  priority: 2,
  weight: 0.1746,
  score: 0.6709,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_018',
  name: 'node_018',
  version: '1.6',
  status: 'recovered',
  priority: 6,
  weight: 0.2148,
  score: 0.3821,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_019',
  name: 'node_019',
  version: '4.6',
  status: 'pending',
  priority: 10,
  weight: 0.3083,
  score: 0.2684,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_020',
  name: 'node_020',
  version: '1.2',
  status: 'degraded',
  priority: 9,
  weight: 0.9645,
  score: 0.8573,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_021',
  name: 'node_021',
  version: '1.8',
  status: 'pending',
  priority: 5,
  weight: 0.9408,
  score: 0.5095,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_022',
  name: 'node_022',
  version: '1.0',
  status: 'failed',
  priority: 2,
  weight: 0.6625,
  score: 0.0446,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_023',
  name: 'node_023',
  version: '4.9',
  status: 'degraded',
  priority: 4,
  weight: 0.1221,
  score: 0.9544,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_024',
  name: 'node_024',
  version: '2.8',
  status: 'active',
  priority: 7,
  weight: 0.3192,
  score: 0.2663,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_025',
  name: 'node_025',
  version: '5.6',
  status: 'completed',
  priority: 5,
  weight: 0.6518,
  score: 0.2597,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_026',
  name: 'node_026',
  version: '5.4',
  status: 'stable',
  priority: 6,
  weight: 0.185,
  score: 0.2604,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_027',
  name: 'node_027',
  version: '2.0',
  status: 'stable',
  priority: 9,
  weight: 0.9243,
  score: 0.8058,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_028',
  name: 'node_028',
  version: '3.4',
  status: 'failed',
  priority: 4,
  weight: 0.2662,
  score: 0.086,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_029',
  name: 'node_029',
  version: '2.9',
  status: 'active',
  priority: 6,
  weight: 0.1872,
  score: 0.3001,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_030',
  name: 'node_030',
  version: '5.9',
  status: 'failed',
  priority: 8,
  weight: 0.6413,
  score: 0.3643,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_031',
  name: 'node_031',
  version: '2.6',
  status: 'completed',
  priority: 7,
  weight: 0.657,
  score: 0.6175,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_032',
  name: 'node_032',
  version: '5.0',
  status: 'stable',
  priority: 4,
  weight: 0.4134,
  score: 0.792,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_033',
  name: 'node_033',
  version: '1.2',
  status: 'recovered',
  priority: 8,
  weight: 0.4991,
  score: 0.4581,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_034',
  name: 'node_034',
  version: '3.4',
  status: 'degraded',
  priority: 2,
  weight: 0.5629,
  score: 0.7601,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_035',
  name: 'node_035',
  version: '3.0',
  status: 'degraded',
  priority: 7,
  weight: 0.7808,
  score: 0.6365,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_036',
  name: 'node_036',
  version: '1.3',
  status: 'recovered',
  priority: 1,
  weight: 0.6835,
  score: 0.1874,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_037',
  name: 'node_037',
  version: '4.0',
  status: 'failed',
  priority: 5,
  weight: 0.1448,
  score: 0.1472,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_038',
  name: 'node_038',
  version: '4.6',
  status: 'active',
  priority: 4,
  weight: 0.3619,
  score: 0.8602,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_01_core_engine_1_039',
  name: 'node_039',
  version: '5.1',
  status: 'failed',
  priority: 10,
  weight: 0.8944,
  score: 0.8687,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: true
});
