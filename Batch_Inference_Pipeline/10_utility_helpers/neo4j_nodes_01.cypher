:param namespace => 'batchinference_01_01';
:param batchSize => 64;
:param threshold => 0.839;
:param maxDepth => 3;
:param timeoutSeconds => 33;
:param region => 'us-west';
:param epoch => 21;
:param version => '4.6.3';

CREATE (n_000:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_000',
  name: 'node_000',
  version: '2.9',
  status: 'recovered',
  priority: 10,
  weight: 0.6858,
  score: 0.8162,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_001',
  name: 'node_001',
  version: '5.1',
  status: 'active',
  priority: 7,
  weight: 0.8055,
  score: 0.0542,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_002',
  name: 'node_002',
  version: '1.5',
  status: 'degraded',
  priority: 1,
  weight: 0.6352,
  score: 0.3142,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_003',
  name: 'node_003',
  version: '4.4',
  status: 'active',
  priority: 1,
  weight: 0.7356,
  score: 0.6123,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_004',
  name: 'node_004',
  version: '4.7',
  status: 'stable',
  priority: 10,
  weight: 0.7657,
  score: 0.5394,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_005',
  name: 'node_005',
  version: '2.1',
  status: 'completed',
  priority: 4,
  weight: 0.6,
  score: 0.6483,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_006',
  name: 'node_006',
  version: '2.8',
  status: 'active',
  priority: 2,
  weight: 0.1627,
  score: 0.6642,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_007',
  name: 'node_007',
  version: '4.3',
  status: 'stable',
  priority: 5,
  weight: 0.167,
  score: 0.8178,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_008',
  name: 'node_008',
  version: '5.9',
  status: 'completed',
  priority: 1,
  weight: 0.2615,
  score: 0.8599,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_009',
  name: 'node_009',
  version: '1.2',
  status: 'degraded',
  priority: 10,
  weight: 0.6561,
  score: 0.6456,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_010',
  name: 'node_010',
  version: '2.0',
  status: 'pending',
  priority: 5,
  weight: 0.6824,
  score: 0.4135,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_011',
  name: 'node_011',
  version: '3.3',
  status: 'active',
  priority: 7,
  weight: 0.5874,
  score: 0.9956,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_012',
  name: 'node_012',
  version: '5.6',
  status: 'completed',
  priority: 3,
  weight: 0.9577,
  score: 0.8951,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_013',
  name: 'node_013',
  version: '5.9',
  status: 'degraded',
  priority: 8,
  weight: 0.4536,
  score: 0.6831,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_014',
  name: 'node_014',
  version: '5.4',
  status: 'active',
  priority: 5,
  weight: 0.5824,
  score: 0.7867,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_015',
  name: 'node_015',
  version: '3.4',
  status: 'pending',
  priority: 3,
  weight: 0.1181,
  score: 0.5421,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_016',
  name: 'node_016',
  version: '2.0',
  status: 'active',
  priority: 3,
  weight: 0.4536,
  score: 0.446,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_017',
  name: 'node_017',
  version: '5.1',
  status: 'completed',
  priority: 8,
  weight: 0.9223,
  score: 0.7403,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_018',
  name: 'node_018',
  version: '5.7',
  status: 'completed',
  priority: 7,
  weight: 0.8653,
  score: 0.9348,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_019',
  name: 'node_019',
  version: '4.0',
  status: 'active',
  priority: 7,
  weight: 0.3726,
  score: 0.1667,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_020',
  name: 'node_020',
  version: '3.6',
  status: 'active',
  priority: 1,
  weight: 0.1843,
  score: 0.9653,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_021',
  name: 'node_021',
  version: '5.5',
  status: 'degraded',
  priority: 3,
  weight: 0.2557,
  score: 0.6178,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_022',
  name: 'node_022',
  version: '5.3',
  status: 'recovered',
  priority: 6,
  weight: 0.4516,
  score: 0.6574,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_023',
  name: 'node_023',
  version: '5.8',
  status: 'recovered',
  priority: 2,
  weight: 0.3123,
  score: 0.3325,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_024',
  name: 'node_024',
  version: '5.5',
  status: 'stable',
  priority: 1,
  weight: 0.4632,
  score: 0.131,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_025',
  name: 'node_025',
  version: '4.8',
  status: 'active',
  priority: 1,
  weight: 0.6332,
  score: 0.6272,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_026',
  name: 'node_026',
  version: '1.3',
  status: 'active',
  priority: 5,
  weight: 0.4471,
  score: 0.3405,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_027',
  name: 'node_027',
  version: '1.8',
  status: 'pending',
  priority: 4,
  weight: 0.3199,
  score: 0.7843,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_028',
  name: 'node_028',
  version: '1.8',
  status: 'active',
  priority: 6,
  weight: 0.9468,
  score: 0.8065,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_029',
  name: 'node_029',
  version: '2.7',
  status: 'stable',
  priority: 3,
  weight: 0.2721,
  score: 0.8575,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_030',
  name: 'node_030',
  version: '4.5',
  status: 'failed',
  priority: 1,
  weight: 0.4801,
  score: 0.9621,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_031',
  name: 'node_031',
  version: '5.5',
  status: 'completed',
  priority: 6,
  weight: 0.7908,
  score: 0.379,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_032',
  name: 'node_032',
  version: '2.3',
  status: 'completed',
  priority: 9,
  weight: 0.2698,
  score: 0.8603,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_033',
  name: 'node_033',
  version: '5.6',
  status: 'degraded',
  priority: 3,
  weight: 0.798,
  score: 0.2481,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_034',
  name: 'node_034',
  version: '1.2',
  status: 'pending',
  priority: 7,
  weight: 0.4561,
  score: 0.3245,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_035',
  name: 'node_035',
  version: '4.6',
  status: 'completed',
  priority: 10,
  weight: 0.9507,
  score: 0.5889,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_036',
  name: 'node_036',
  version: '3.1',
  status: 'stable',
  priority: 8,
  weight: 0.7665,
  score: 0.6293,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_037',
  name: 'node_037',
  version: '1.2',
  status: 'pending',
  priority: 3,
  weight: 0.2561,
  score: 0.7403,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_038',
  name: 'node_038',
  version: '5.7',
  status: 'completed',
  priority: 4,
  weight: 0.2372,
  score: 0.1225,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:BatchInference:Node {
  identifier: 'batchinference_10_utility_helpers_1_039',
  name: 'node_039',
  version: '4.7',
  status: 'stable',
  priority: 8,
  weight: 0.625,
  score: 0.7934,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: false
});
