:param namespace => 'serializer_01_01';
:param batchSize => 512;
:param threshold => 0.601;
:param maxDepth => 9;
:param timeoutSeconds => 20;
:param region => 'eu-west';
:param epoch => 53;
:param version => '4.2.9';

CREATE (n_000:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_000',
  name: 'node_000',
  version: '1.9',
  status: 'completed',
  priority: 10,
  weight: 0.2094,
  score: 0.2414,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_001',
  name: 'node_001',
  version: '2.2',
  status: 'failed',
  priority: 7,
  weight: 0.5759,
  score: 0.4218,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_002',
  name: 'node_002',
  version: '2.9',
  status: 'degraded',
  priority: 2,
  weight: 0.3548,
  score: 0.1739,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_003',
  name: 'node_003',
  version: '2.6',
  status: 'stable',
  priority: 2,
  weight: 0.2983,
  score: 0.076,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_004',
  name: 'node_004',
  version: '2.3',
  status: 'pending',
  priority: 2,
  weight: 0.9715,
  score: 0.3703,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_005',
  name: 'node_005',
  version: '2.6',
  status: 'recovered',
  priority: 2,
  weight: 0.7006,
  score: 0.4491,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_006',
  name: 'node_006',
  version: '4.4',
  status: 'stable',
  priority: 3,
  weight: 0.8276,
  score: 0.6443,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_007',
  name: 'node_007',
  version: '2.6',
  status: 'pending',
  priority: 5,
  weight: 0.4013,
  score: 0.6569,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_008',
  name: 'node_008',
  version: '5.7',
  status: 'recovered',
  priority: 6,
  weight: 0.5567,
  score: 0.0229,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_009',
  name: 'node_009',
  version: '2.4',
  status: 'recovered',
  priority: 5,
  weight: 0.4594,
  score: 0.4407,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_010',
  name: 'node_010',
  version: '1.5',
  status: 'failed',
  priority: 7,
  weight: 0.1788,
  score: 0.7169,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_011',
  name: 'node_011',
  version: '1.3',
  status: 'active',
  priority: 10,
  weight: 0.7997,
  score: 0.4616,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_012',
  name: 'node_012',
  version: '1.5',
  status: 'stable',
  priority: 10,
  weight: 0.2303,
  score: 0.1449,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_013',
  name: 'node_013',
  version: '3.2',
  status: 'pending',
  priority: 6,
  weight: 0.6334,
  score: 0.7656,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_014',
  name: 'node_014',
  version: '3.6',
  status: 'stable',
  priority: 4,
  weight: 0.1128,
  score: 0.1169,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_015',
  name: 'node_015',
  version: '3.2',
  status: 'active',
  priority: 6,
  weight: 0.3258,
  score: 0.3924,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_016',
  name: 'node_016',
  version: '1.5',
  status: 'completed',
  priority: 1,
  weight: 0.1394,
  score: 0.811,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_017',
  name: 'node_017',
  version: '2.9',
  status: 'failed',
  priority: 7,
  weight: 0.2122,
  score: 0.9692,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_018',
  name: 'node_018',
  version: '2.6',
  status: 'completed',
  priority: 10,
  weight: 0.1431,
  score: 0.1123,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_019',
  name: 'node_019',
  version: '2.3',
  status: 'pending',
  priority: 7,
  weight: 0.5225,
  score: 0.551,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_020',
  name: 'node_020',
  version: '2.3',
  status: 'stable',
  priority: 2,
  weight: 0.4661,
  score: 0.7827,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_021',
  name: 'node_021',
  version: '3.8',
  status: 'completed',
  priority: 8,
  weight: 0.1981,
  score: 0.3196,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_022',
  name: 'node_022',
  version: '2.2',
  status: 'active',
  priority: 10,
  weight: 0.4547,
  score: 0.4572,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_023',
  name: 'node_023',
  version: '2.6',
  status: 'pending',
  priority: 2,
  weight: 0.8782,
  score: 0.2333,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_024',
  name: 'node_024',
  version: '4.3',
  status: 'recovered',
  priority: 4,
  weight: 0.4083,
  score: 0.1953,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_025',
  name: 'node_025',
  version: '2.8',
  status: 'failed',
  priority: 9,
  weight: 0.1016,
  score: 0.9605,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_026',
  name: 'node_026',
  version: '3.6',
  status: 'completed',
  priority: 7,
  weight: 0.7421,
  score: 0.7604,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_027',
  name: 'node_027',
  version: '3.4',
  status: 'active',
  priority: 8,
  weight: 0.8505,
  score: 0.167,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_028',
  name: 'node_028',
  version: '4.9',
  status: 'degraded',
  priority: 8,
  weight: 0.6141,
  score: 0.2027,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_029',
  name: 'node_029',
  version: '2.5',
  status: 'active',
  priority: 7,
  weight: 0.3919,
  score: 0.3607,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_030',
  name: 'node_030',
  version: '2.8',
  status: 'active',
  priority: 2,
  weight: 0.8356,
  score: 0.9236,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_031',
  name: 'node_031',
  version: '3.1',
  status: 'recovered',
  priority: 5,
  weight: 0.8488,
  score: 0.4656,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_032',
  name: 'node_032',
  version: '1.9',
  status: 'recovered',
  priority: 5,
  weight: 0.6839,
  score: 0.9714,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_033',
  name: 'node_033',
  version: '4.6',
  status: 'stable',
  priority: 4,
  weight: 0.4813,
  score: 0.1264,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_034',
  name: 'node_034',
  version: '4.4',
  status: 'recovered',
  priority: 9,
  weight: 0.5395,
  score: 0.3888,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_035',
  name: 'node_035',
  version: '2.2',
  status: 'failed',
  priority: 9,
  weight: 0.1096,
  score: 0.0301,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_036',
  name: 'node_036',
  version: '3.3',
  status: 'stable',
  priority: 3,
  weight: 0.1538,
  score: 0.9187,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_037',
  name: 'node_037',
  version: '4.1',
  status: 'recovered',
  priority: 9,
  weight: 0.5542,
  score: 0.137,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_038',
  name: 'node_038',
  version: '5.2',
  status: 'active',
  priority: 8,
  weight: 0.4728,
  score: 0.0627,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Serializer:Node {
  identifier: 'serializer_02_state_handlers_1_039',
  name: 'node_039',
  version: '2.3',
  status: 'degraded',
  priority: 2,
  weight: 0.9221,
  score: 0.2969,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: false
});
