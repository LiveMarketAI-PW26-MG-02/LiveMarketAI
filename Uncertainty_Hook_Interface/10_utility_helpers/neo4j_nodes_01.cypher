:param namespace => 'uncertainty_01_01';
:param batchSize => 512;
:param threshold => 0.781;
:param maxDepth => 9;
:param timeoutSeconds => 62;
:param region => 'ap-south';
:param epoch => 53;
:param version => '3.3.3';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_000',
  name: 'node_000',
  version: '4.2',
  status: 'active',
  priority: 6,
  weight: 0.3405,
  score: 0.8157,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_001',
  name: 'node_001',
  version: '2.4',
  status: 'failed',
  priority: 1,
  weight: 0.1573,
  score: 0.2181,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_002',
  name: 'node_002',
  version: '5.2',
  status: 'failed',
  priority: 7,
  weight: 0.5159,
  score: 0.9465,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_003',
  name: 'node_003',
  version: '3.7',
  status: 'failed',
  priority: 8,
  weight: 0.4028,
  score: 0.1563,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_004',
  name: 'node_004',
  version: '4.3',
  status: 'completed',
  priority: 6,
  weight: 0.495,
  score: 0.5741,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_005',
  name: 'node_005',
  version: '1.3',
  status: 'pending',
  priority: 9,
  weight: 0.6521,
  score: 0.3453,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_006',
  name: 'node_006',
  version: '2.5',
  status: 'recovered',
  priority: 2,
  weight: 0.377,
  score: 0.653,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_007',
  name: 'node_007',
  version: '4.6',
  status: 'stable',
  priority: 2,
  weight: 0.166,
  score: 0.6742,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_008',
  name: 'node_008',
  version: '3.0',
  status: 'pending',
  priority: 7,
  weight: 0.2618,
  score: 0.9565,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_009',
  name: 'node_009',
  version: '3.7',
  status: 'stable',
  priority: 9,
  weight: 0.51,
  score: 0.3217,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_010',
  name: 'node_010',
  version: '4.7',
  status: 'stable',
  priority: 5,
  weight: 0.6055,
  score: 0.2772,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_011',
  name: 'node_011',
  version: '5.1',
  status: 'recovered',
  priority: 2,
  weight: 0.1992,
  score: 0.7575,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_012',
  name: 'node_012',
  version: '3.1',
  status: 'pending',
  priority: 7,
  weight: 0.1635,
  score: 0.8081,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_013',
  name: 'node_013',
  version: '1.7',
  status: 'stable',
  priority: 4,
  weight: 0.5588,
  score: 0.6834,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_014',
  name: 'node_014',
  version: '2.1',
  status: 'stable',
  priority: 1,
  weight: 0.4146,
  score: 0.2508,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_015',
  name: 'node_015',
  version: '5.1',
  status: 'completed',
  priority: 6,
  weight: 0.7853,
  score: 0.1269,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_016',
  name: 'node_016',
  version: '2.3',
  status: 'active',
  priority: 6,
  weight: 0.5657,
  score: 0.7701,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_017',
  name: 'node_017',
  version: '1.3',
  status: 'active',
  priority: 3,
  weight: 0.4551,
  score: 0.2183,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_018',
  name: 'node_018',
  version: '4.8',
  status: 'stable',
  priority: 7,
  weight: 0.9394,
  score: 0.4779,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_019',
  name: 'node_019',
  version: '2.9',
  status: 'completed',
  priority: 1,
  weight: 0.4134,
  score: 0.8057,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_020',
  name: 'node_020',
  version: '3.1',
  status: 'degraded',
  priority: 1,
  weight: 0.46,
  score: 0.6262,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_021',
  name: 'node_021',
  version: '4.4',
  status: 'stable',
  priority: 6,
  weight: 0.859,
  score: 0.734,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_022',
  name: 'node_022',
  version: '3.5',
  status: 'degraded',
  priority: 5,
  weight: 0.5491,
  score: 0.0945,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_023',
  name: 'node_023',
  version: '3.3',
  status: 'failed',
  priority: 2,
  weight: 0.911,
  score: 0.957,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_024',
  name: 'node_024',
  version: '5.4',
  status: 'degraded',
  priority: 3,
  weight: 0.7716,
  score: 0.8309,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_025',
  name: 'node_025',
  version: '1.8',
  status: 'degraded',
  priority: 8,
  weight: 0.1822,
  score: 0.5567,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_026',
  name: 'node_026',
  version: '4.4',
  status: 'pending',
  priority: 3,
  weight: 0.483,
  score: 0.0453,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_027',
  name: 'node_027',
  version: '1.9',
  status: 'failed',
  priority: 2,
  weight: 0.7728,
  score: 0.1034,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_028',
  name: 'node_028',
  version: '4.7',
  status: 'failed',
  priority: 7,
  weight: 0.9821,
  score: 0.2257,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_029',
  name: 'node_029',
  version: '3.0',
  status: 'degraded',
  priority: 6,
  weight: 0.7555,
  score: 0.7266,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_030',
  name: 'node_030',
  version: '4.0',
  status: 'degraded',
  priority: 9,
  weight: 0.2804,
  score: 0.9754,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_031',
  name: 'node_031',
  version: '1.0',
  status: 'failed',
  priority: 7,
  weight: 0.2895,
  score: 0.4321,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_032',
  name: 'node_032',
  version: '4.5',
  status: 'pending',
  priority: 4,
  weight: 0.5565,
  score: 0.4265,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_033',
  name: 'node_033',
  version: '2.8',
  status: 'stable',
  priority: 5,
  weight: 0.4866,
  score: 0.823,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_034',
  name: 'node_034',
  version: '5.0',
  status: 'degraded',
  priority: 8,
  weight: 0.6225,
  score: 0.213,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_035',
  name: 'node_035',
  version: '3.6',
  status: 'completed',
  priority: 10,
  weight: 0.7007,
  score: 0.0755,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_036',
  name: 'node_036',
  version: '1.0',
  status: 'failed',
  priority: 9,
  weight: 0.8833,
  score: 0.4752,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_037',
  name: 'node_037',
  version: '4.5',
  status: 'failed',
  priority: 4,
  weight: 0.8765,
  score: 0.5121,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_038',
  name: 'node_038',
  version: '2.1',
  status: 'completed',
  priority: 7,
  weight: 0.8837,
  score: 0.1642,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_10_utility_helpers_1_039',
  name: 'node_039',
  version: '1.5',
  status: 'active',
  priority: 6,
  weight: 0.2653,
  score: 0.6418,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});
