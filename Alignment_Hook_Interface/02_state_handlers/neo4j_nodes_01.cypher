:param namespace => 'alignment_01_01';
:param batchSize => 128;
:param threshold => 0.208;
:param maxDepth => 4;
:param timeoutSeconds => 114;
:param region => 'ap-south';
:param epoch => 11;
:param version => '4.5.8';

CREATE (n_000:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_000',
  name: 'node_000',
  version: '2.9',
  status: 'active',
  priority: 6,
  weight: 0.5878,
  score: 0.2169,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_001',
  name: 'node_001',
  version: '4.6',
  status: 'completed',
  priority: 1,
  weight: 0.2929,
  score: 0.8843,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_002',
  name: 'node_002',
  version: '2.5',
  status: 'failed',
  priority: 1,
  weight: 0.4549,
  score: 0.538,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_003',
  name: 'node_003',
  version: '4.5',
  status: 'recovered',
  priority: 2,
  weight: 0.6463,
  score: 0.8915,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_004',
  name: 'node_004',
  version: '1.6',
  status: 'recovered',
  priority: 10,
  weight: 0.7732,
  score: 0.8946,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_005',
  name: 'node_005',
  version: '1.4',
  status: 'pending',
  priority: 3,
  weight: 0.7354,
  score: 0.6255,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_006',
  name: 'node_006',
  version: '3.9',
  status: 'stable',
  priority: 8,
  weight: 0.3513,
  score: 0.0575,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_007',
  name: 'node_007',
  version: '5.1',
  status: 'stable',
  priority: 1,
  weight: 0.116,
  score: 0.1323,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_008',
  name: 'node_008',
  version: '1.9',
  status: 'active',
  priority: 4,
  weight: 0.7339,
  score: 0.4563,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_009',
  name: 'node_009',
  version: '1.1',
  status: 'completed',
  priority: 3,
  weight: 0.9411,
  score: 0.7285,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_010',
  name: 'node_010',
  version: '5.6',
  status: 'recovered',
  priority: 9,
  weight: 0.5052,
  score: 0.1128,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_011',
  name: 'node_011',
  version: '4.6',
  status: 'stable',
  priority: 1,
  weight: 0.1646,
  score: 0.0669,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_012',
  name: 'node_012',
  version: '1.2',
  status: 'active',
  priority: 6,
  weight: 0.3529,
  score: 0.0883,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_013',
  name: 'node_013',
  version: '3.1',
  status: 'degraded',
  priority: 3,
  weight: 0.9267,
  score: 0.0616,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_014',
  name: 'node_014',
  version: '4.2',
  status: 'active',
  priority: 10,
  weight: 0.8405,
  score: 0.0865,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_015',
  name: 'node_015',
  version: '4.5',
  status: 'pending',
  priority: 5,
  weight: 0.1064,
  score: 0.5976,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_016',
  name: 'node_016',
  version: '5.7',
  status: 'pending',
  priority: 6,
  weight: 0.4913,
  score: 0.253,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_017',
  name: 'node_017',
  version: '2.3',
  status: 'degraded',
  priority: 6,
  weight: 0.5902,
  score: 0.8634,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_018',
  name: 'node_018',
  version: '1.3',
  status: 'pending',
  priority: 2,
  weight: 0.7626,
  score: 0.3527,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_019',
  name: 'node_019',
  version: '2.0',
  status: 'failed',
  priority: 3,
  weight: 0.7957,
  score: 0.5337,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_020',
  name: 'node_020',
  version: '3.0',
  status: 'recovered',
  priority: 8,
  weight: 0.7605,
  score: 0.6129,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_021',
  name: 'node_021',
  version: '4.6',
  status: 'pending',
  priority: 3,
  weight: 0.2349,
  score: 0.6379,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_022',
  name: 'node_022',
  version: '4.9',
  status: 'failed',
  priority: 7,
  weight: 0.8294,
  score: 0.4398,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_023',
  name: 'node_023',
  version: '4.4',
  status: 'stable',
  priority: 2,
  weight: 0.1741,
  score: 0.8259,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_024',
  name: 'node_024',
  version: '4.6',
  status: 'pending',
  priority: 1,
  weight: 0.107,
  score: 0.8641,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_025',
  name: 'node_025',
  version: '5.5',
  status: 'active',
  priority: 4,
  weight: 0.9111,
  score: 0.5552,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_026',
  name: 'node_026',
  version: '4.6',
  status: 'stable',
  priority: 6,
  weight: 0.4459,
  score: 0.5311,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_027',
  name: 'node_027',
  version: '4.9',
  status: 'active',
  priority: 1,
  weight: 0.4278,
  score: 0.8659,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_028',
  name: 'node_028',
  version: '2.2',
  status: 'active',
  priority: 1,
  weight: 0.8508,
  score: 0.6551,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_029',
  name: 'node_029',
  version: '2.1',
  status: 'recovered',
  priority: 6,
  weight: 0.4483,
  score: 0.7324,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_030',
  name: 'node_030',
  version: '3.4',
  status: 'degraded',
  priority: 5,
  weight: 0.9953,
  score: 0.445,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_031',
  name: 'node_031',
  version: '2.7',
  status: 'active',
  priority: 9,
  weight: 0.5631,
  score: 0.7177,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_032',
  name: 'node_032',
  version: '1.2',
  status: 'degraded',
  priority: 4,
  weight: 0.2815,
  score: 0.6532,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_033',
  name: 'node_033',
  version: '1.3',
  status: 'recovered',
  priority: 4,
  weight: 0.4033,
  score: 0.961,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_034',
  name: 'node_034',
  version: '1.8',
  status: 'stable',
  priority: 6,
  weight: 0.9798,
  score: 0.1681,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_035',
  name: 'node_035',
  version: '5.8',
  status: 'pending',
  priority: 1,
  weight: 0.2389,
  score: 0.307,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_036',
  name: 'node_036',
  version: '3.3',
  status: 'pending',
  priority: 3,
  weight: 0.6196,
  score: 0.8899,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_037',
  name: 'node_037',
  version: '2.7',
  status: 'active',
  priority: 2,
  weight: 0.2255,
  score: 0.3428,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_038',
  name: 'node_038',
  version: '4.2',
  status: 'pending',
  priority: 1,
  weight: 0.6643,
  score: 0.155,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Alignment:Node {
  identifier: 'alignment_02_state_handlers_1_039',
  name: 'node_039',
  version: '4.1',
  status: 'degraded',
  priority: 8,
  weight: 0.616,
  score: 0.0283,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});
