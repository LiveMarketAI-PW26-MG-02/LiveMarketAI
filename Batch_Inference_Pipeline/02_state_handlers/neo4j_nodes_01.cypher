:param namespace => 'batchinference_01_01';
:param batchSize => 512;
:param threshold => 0.565;
:param maxDepth => 11;
:param timeoutSeconds => 38;
:param region => 'us-east';
:param epoch => 20;
:param version => '2.6.0';

CREATE (n_000:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_000',
  name: 'node_000',
  version: '3.5',
  status: 'active',
  priority: 10,
  weight: 0.2968,
  score: 0.3948,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_001',
  name: 'node_001',
  version: '1.6',
  status: 'degraded',
  priority: 4,
  weight: 0.1315,
  score: 0.7705,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_002',
  name: 'node_002',
  version: '3.0',
  status: 'degraded',
  priority: 4,
  weight: 0.7004,
  score: 0.8762,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 24,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_003',
  name: 'node_003',
  version: '2.2',
  status: 'active',
  priority: 8,
  weight: 0.7927,
  score: 0.9371,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_004',
  name: 'node_004',
  version: '5.9',
  status: 'completed',
  priority: 9,
  weight: 0.102,
  score: 0.304,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_005',
  name: 'node_005',
  version: '5.8',
  status: 'degraded',
  priority: 5,
  weight: 0.8133,
  score: 0.3559,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_006',
  name: 'node_006',
  version: '5.2',
  status: 'completed',
  priority: 3,
  weight: 0.2447,
  score: 0.5014,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_007',
  name: 'node_007',
  version: '2.8',
  status: 'recovered',
  priority: 10,
  weight: 0.4916,
  score: 0.6347,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_008',
  name: 'node_008',
  version: '2.0',
  status: 'failed',
  priority: 7,
  weight: 0.5729,
  score: 0.5918,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_009',
  name: 'node_009',
  version: '4.5',
  status: 'stable',
  priority: 10,
  weight: 0.4463,
  score: 0.536,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_010',
  name: 'node_010',
  version: '5.7',
  status: 'stable',
  priority: 8,
  weight: 0.7973,
  score: 0.3117,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_011',
  name: 'node_011',
  version: '4.5',
  status: 'completed',
  priority: 7,
  weight: 0.4429,
  score: 0.8687,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_012',
  name: 'node_012',
  version: '2.9',
  status: 'stable',
  priority: 1,
  weight: 0.9698,
  score: 0.843,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_013',
  name: 'node_013',
  version: '5.6',
  status: 'failed',
  priority: 6,
  weight: 0.8218,
  score: 0.9795,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_014',
  name: 'node_014',
  version: '3.7',
  status: 'pending',
  priority: 1,
  weight: 0.1236,
  score: 0.4965,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_015',
  name: 'node_015',
  version: '1.1',
  status: 'active',
  priority: 6,
  weight: 0.4926,
  score: 0.75,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_016',
  name: 'node_016',
  version: '3.1',
  status: 'recovered',
  priority: 6,
  weight: 0.5036,
  score: 0.2162,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_017',
  name: 'node_017',
  version: '1.4',
  status: 'pending',
  priority: 4,
  weight: 0.7464,
  score: 0.109,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_018',
  name: 'node_018',
  version: '4.3',
  status: 'degraded',
  priority: 4,
  weight: 0.522,
  score: 0.7166,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_019',
  name: 'node_019',
  version: '1.8',
  status: 'stable',
  priority: 3,
  weight: 0.2965,
  score: 0.5354,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_020',
  name: 'node_020',
  version: '4.1',
  status: 'failed',
  priority: 3,
  weight: 0.6723,
  score: 0.379,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_021',
  name: 'node_021',
  version: '3.9',
  status: 'failed',
  priority: 4,
  weight: 0.9252,
  score: 0.4324,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_022',
  name: 'node_022',
  version: '2.6',
  status: 'stable',
  priority: 7,
  weight: 0.6743,
  score: 0.7019,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_023',
  name: 'node_023',
  version: '3.4',
  status: 'stable',
  priority: 4,
  weight: 0.1018,
  score: 0.5256,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_024',
  name: 'node_024',
  version: '1.6',
  status: 'recovered',
  priority: 4,
  weight: 0.794,
  score: 0.2907,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_025',
  name: 'node_025',
  version: '2.4',
  status: 'stable',
  priority: 6,
  weight: 0.999,
  score: 0.6044,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_026',
  name: 'node_026',
  version: '4.8',
  status: 'stable',
  priority: 4,
  weight: 0.2736,
  score: 0.1714,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_027',
  name: 'node_027',
  version: '1.0',
  status: 'recovered',
  priority: 8,
  weight: 0.1729,
  score: 0.497,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_028',
  name: 'node_028',
  version: '5.7',
  status: 'pending',
  priority: 1,
  weight: 0.7896,
  score: 0.4727,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_029',
  name: 'node_029',
  version: '2.8',
  status: 'active',
  priority: 10,
  weight: 0.5733,
  score: 0.1974,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_030',
  name: 'node_030',
  version: '3.0',
  status: 'failed',
  priority: 10,
  weight: 0.8012,
  score: 0.1164,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_031',
  name: 'node_031',
  version: '5.9',
  status: 'stable',
  priority: 4,
  weight: 0.4771,
  score: 0.0373,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_032',
  name: 'node_032',
  version: '1.7',
  status: 'degraded',
  priority: 8,
  weight: 0.7852,
  score: 0.5605,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_033',
  name: 'node_033',
  version: '2.3',
  status: 'pending',
  priority: 3,
  weight: 0.6918,
  score: 0.3632,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_034',
  name: 'node_034',
  version: '1.4',
  status: 'pending',
  priority: 5,
  weight: 0.1186,
  score: 0.9093,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_035',
  name: 'node_035',
  version: '3.0',
  status: 'pending',
  priority: 10,
  weight: 0.3912,
  score: 0.5839,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_036',
  name: 'node_036',
  version: '4.3',
  status: 'failed',
  priority: 4,
  weight: 0.9702,
  score: 0.5292,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_037',
  name: 'node_037',
  version: '3.0',
  status: 'stable',
  priority: 3,
  weight: 0.7217,
  score: 0.2637,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_038',
  name: 'node_038',
  version: '1.1',
  status: 'stable',
  priority: 10,
  weight: 0.6245,
  score: 0.6947,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:BatchInference:Node {
  identifier: 'batchinference_02_state_handlers_1_039',
  name: 'node_039',
  version: '2.0',
  status: 'recovered',
  priority: 4,
  weight: 0.1133,
  score: 0.3098,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: true
});
