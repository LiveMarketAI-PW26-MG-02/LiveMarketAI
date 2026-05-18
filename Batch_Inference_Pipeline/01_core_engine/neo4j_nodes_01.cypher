:param namespace => 'batchinference_01_01';
:param batchSize => 128;
:param threshold => 0.353;
:param maxDepth => 7;
:param timeoutSeconds => 112;
:param region => 'ap-south';
:param epoch => 44;
:param version => '5.2.7';

CREATE (n_000:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_000',
  name: 'node_000',
  version: '4.0',
  status: 'degraded',
  priority: 4,
  weight: 0.9619,
  score: 0.5117,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_001',
  name: 'node_001',
  version: '5.1',
  status: 'completed',
  priority: 9,
  weight: 0.612,
  score: 0.826,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_002',
  name: 'node_002',
  version: '4.7',
  status: 'stable',
  priority: 6,
  weight: 0.6113,
  score: 0.2161,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_003',
  name: 'node_003',
  version: '1.6',
  status: 'degraded',
  priority: 6,
  weight: 0.5332,
  score: 0.6175,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_004',
  name: 'node_004',
  version: '1.5',
  status: 'stable',
  priority: 2,
  weight: 0.2889,
  score: 0.3388,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_005',
  name: 'node_005',
  version: '1.2',
  status: 'stable',
  priority: 10,
  weight: 0.8024,
  score: 0.7839,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_006',
  name: 'node_006',
  version: '5.1',
  status: 'pending',
  priority: 4,
  weight: 0.4731,
  score: 0.4605,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_007',
  name: 'node_007',
  version: '2.3',
  status: 'active',
  priority: 1,
  weight: 0.1276,
  score: 0.4214,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_008',
  name: 'node_008',
  version: '2.5',
  status: 'active',
  priority: 8,
  weight: 0.9584,
  score: 0.1994,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_009',
  name: 'node_009',
  version: '5.2',
  status: 'stable',
  priority: 5,
  weight: 0.7707,
  score: 0.3736,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_010',
  name: 'node_010',
  version: '4.6',
  status: 'pending',
  priority: 2,
  weight: 0.7631,
  score: 0.9767,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_011',
  name: 'node_011',
  version: '2.7',
  status: 'stable',
  priority: 6,
  weight: 0.3698,
  score: 0.7716,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_012',
  name: 'node_012',
  version: '5.1',
  status: 'degraded',
  priority: 4,
  weight: 0.7968,
  score: 0.3288,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_013',
  name: 'node_013',
  version: '1.0',
  status: 'failed',
  priority: 8,
  weight: 0.2341,
  score: 0.4235,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_014',
  name: 'node_014',
  version: '1.3',
  status: 'active',
  priority: 9,
  weight: 0.892,
  score: 0.7285,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_015',
  name: 'node_015',
  version: '4.3',
  status: 'degraded',
  priority: 8,
  weight: 0.4315,
  score: 0.3845,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_016',
  name: 'node_016',
  version: '1.0',
  status: 'recovered',
  priority: 10,
  weight: 0.1325,
  score: 0.6817,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_017',
  name: 'node_017',
  version: '2.0',
  status: 'stable',
  priority: 4,
  weight: 0.6051,
  score: 0.7827,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_018',
  name: 'node_018',
  version: '4.0',
  status: 'pending',
  priority: 2,
  weight: 0.9519,
  score: 0.8123,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_019',
  name: 'node_019',
  version: '2.6',
  status: 'stable',
  priority: 8,
  weight: 0.8793,
  score: 0.6672,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_020',
  name: 'node_020',
  version: '3.1',
  status: 'stable',
  priority: 9,
  weight: 0.7457,
  score: 0.6176,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_021',
  name: 'node_021',
  version: '2.8',
  status: 'recovered',
  priority: 2,
  weight: 0.4249,
  score: 0.1736,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_022',
  name: 'node_022',
  version: '4.2',
  status: 'failed',
  priority: 3,
  weight: 0.193,
  score: 0.9051,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_023',
  name: 'node_023',
  version: '2.8',
  status: 'stable',
  priority: 1,
  weight: 0.5153,
  score: 0.362,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_024',
  name: 'node_024',
  version: '1.9',
  status: 'stable',
  priority: 2,
  weight: 0.7152,
  score: 0.3145,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_025',
  name: 'node_025',
  version: '3.8',
  status: 'recovered',
  priority: 2,
  weight: 0.5498,
  score: 0.1143,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_026',
  name: 'node_026',
  version: '5.1',
  status: 'pending',
  priority: 7,
  weight: 0.7139,
  score: 0.3539,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_027',
  name: 'node_027',
  version: '2.4',
  status: 'active',
  priority: 2,
  weight: 0.1018,
  score: 0.3909,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_028',
  name: 'node_028',
  version: '4.2',
  status: 'pending',
  priority: 7,
  weight: 0.9885,
  score: 0.2227,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_029',
  name: 'node_029',
  version: '5.7',
  status: 'recovered',
  priority: 2,
  weight: 0.6058,
  score: 0.4544,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_030',
  name: 'node_030',
  version: '2.3',
  status: 'stable',
  priority: 2,
  weight: 0.7572,
  score: 0.8546,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_031',
  name: 'node_031',
  version: '1.7',
  status: 'degraded',
  priority: 1,
  weight: 0.363,
  score: 0.9583,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_032',
  name: 'node_032',
  version: '4.5',
  status: 'degraded',
  priority: 3,
  weight: 0.4897,
  score: 0.3474,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_033',
  name: 'node_033',
  version: '5.6',
  status: 'pending',
  priority: 3,
  weight: 0.738,
  score: 0.0904,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_034',
  name: 'node_034',
  version: '5.1',
  status: 'completed',
  priority: 3,
  weight: 0.4052,
  score: 0.3536,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_035',
  name: 'node_035',
  version: '2.3',
  status: 'degraded',
  priority: 9,
  weight: 0.7405,
  score: 0.6709,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_036',
  name: 'node_036',
  version: '5.7',
  status: 'degraded',
  priority: 5,
  weight: 0.1814,
  score: 0.0408,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_037',
  name: 'node_037',
  version: '2.6',
  status: 'recovered',
  priority: 5,
  weight: 0.2521,
  score: 0.0677,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_038',
  name: 'node_038',
  version: '3.8',
  status: 'active',
  priority: 3,
  weight: 0.2839,
  score: 0.9762,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BatchInference:Node {
  identifier: 'batchinference_01_core_engine_1_039',
  name: 'node_039',
  version: '2.6',
  status: 'pending',
  priority: 7,
  weight: 0.6425,
  score: 0.8252,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: true
});
