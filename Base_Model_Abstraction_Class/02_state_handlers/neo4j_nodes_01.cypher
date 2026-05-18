:param namespace => 'basemodel_01_01';
:param batchSize => 128;
:param threshold => 0.533;
:param maxDepth => 10;
:param timeoutSeconds => 34;
:param region => 'ap-south';
:param epoch => 47;
:param version => '4.0.6';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_000',
  name: 'node_000',
  version: '2.6',
  status: 'active',
  priority: 7,
  weight: 0.1314,
  score: 0.0626,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_001',
  name: 'node_001',
  version: '5.5',
  status: 'completed',
  priority: 5,
  weight: 0.4015,
  score: 0.9538,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_002',
  name: 'node_002',
  version: '3.4',
  status: 'active',
  priority: 10,
  weight: 0.9248,
  score: 0.634,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_003',
  name: 'node_003',
  version: '4.6',
  status: 'stable',
  priority: 5,
  weight: 0.9222,
  score: 0.8148,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_004',
  name: 'node_004',
  version: '2.9',
  status: 'pending',
  priority: 6,
  weight: 0.8751,
  score: 0.4608,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_005',
  name: 'node_005',
  version: '2.3',
  status: 'failed',
  priority: 2,
  weight: 0.6846,
  score: 0.4817,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_006',
  name: 'node_006',
  version: '1.4',
  status: 'recovered',
  priority: 2,
  weight: 0.2875,
  score: 0.4211,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_007',
  name: 'node_007',
  version: '4.9',
  status: 'degraded',
  priority: 4,
  weight: 0.7732,
  score: 0.847,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_008',
  name: 'node_008',
  version: '3.4',
  status: 'degraded',
  priority: 5,
  weight: 0.2793,
  score: 0.2474,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_009',
  name: 'node_009',
  version: '3.1',
  status: 'failed',
  priority: 5,
  weight: 0.9932,
  score: 0.5073,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_010',
  name: 'node_010',
  version: '1.0',
  status: 'failed',
  priority: 4,
  weight: 0.8565,
  score: 0.9144,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_011',
  name: 'node_011',
  version: '2.9',
  status: 'stable',
  priority: 10,
  weight: 0.2747,
  score: 0.0751,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_012',
  name: 'node_012',
  version: '1.1',
  status: 'degraded',
  priority: 10,
  weight: 0.7387,
  score: 0.3497,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_013',
  name: 'node_013',
  version: '2.4',
  status: 'active',
  priority: 10,
  weight: 0.759,
  score: 0.914,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_014',
  name: 'node_014',
  version: '2.9',
  status: 'completed',
  priority: 2,
  weight: 0.2831,
  score: 0.7953,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_015',
  name: 'node_015',
  version: '4.8',
  status: 'pending',
  priority: 9,
  weight: 0.182,
  score: 0.1637,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_016',
  name: 'node_016',
  version: '4.0',
  status: 'completed',
  priority: 10,
  weight: 0.8953,
  score: 0.4141,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_017',
  name: 'node_017',
  version: '4.3',
  status: 'active',
  priority: 7,
  weight: 0.9115,
  score: 0.4238,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_018',
  name: 'node_018',
  version: '2.2',
  status: 'active',
  priority: 1,
  weight: 0.5964,
  score: 0.6407,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_019',
  name: 'node_019',
  version: '5.2',
  status: 'pending',
  priority: 6,
  weight: 0.355,
  score: 0.5212,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_020',
  name: 'node_020',
  version: '3.2',
  status: 'stable',
  priority: 1,
  weight: 0.978,
  score: 0.4827,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_021',
  name: 'node_021',
  version: '5.2',
  status: 'degraded',
  priority: 4,
  weight: 0.6589,
  score: 0.6147,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_022',
  name: 'node_022',
  version: '1.6',
  status: 'recovered',
  priority: 3,
  weight: 0.4452,
  score: 0.1231,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_023',
  name: 'node_023',
  version: '3.1',
  status: 'failed',
  priority: 10,
  weight: 0.5102,
  score: 0.849,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_024',
  name: 'node_024',
  version: '4.6',
  status: 'degraded',
  priority: 6,
  weight: 0.5021,
  score: 0.4384,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_025',
  name: 'node_025',
  version: '2.7',
  status: 'stable',
  priority: 10,
  weight: 0.802,
  score: 0.4583,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_026',
  name: 'node_026',
  version: '2.5',
  status: 'failed',
  priority: 6,
  weight: 0.1825,
  score: 0.442,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_027',
  name: 'node_027',
  version: '1.5',
  status: 'stable',
  priority: 9,
  weight: 0.172,
  score: 0.7521,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_028',
  name: 'node_028',
  version: '2.2',
  status: 'failed',
  priority: 5,
  weight: 0.961,
  score: 0.916,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_029',
  name: 'node_029',
  version: '2.5',
  status: 'recovered',
  priority: 5,
  weight: 0.9146,
  score: 0.4564,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_030',
  name: 'node_030',
  version: '5.8',
  status: 'pending',
  priority: 6,
  weight: 0.435,
  score: 0.1989,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_031',
  name: 'node_031',
  version: '4.2',
  status: 'stable',
  priority: 5,
  weight: 0.2036,
  score: 0.5307,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_032',
  name: 'node_032',
  version: '3.8',
  status: 'degraded',
  priority: 7,
  weight: 0.7641,
  score: 0.3715,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_033',
  name: 'node_033',
  version: '3.1',
  status: 'failed',
  priority: 4,
  weight: 0.2591,
  score: 0.7436,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_034',
  name: 'node_034',
  version: '5.5',
  status: 'degraded',
  priority: 1,
  weight: 0.7724,
  score: 0.2216,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_035',
  name: 'node_035',
  version: '5.5',
  status: 'active',
  priority: 3,
  weight: 0.5396,
  score: 0.6125,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_036',
  name: 'node_036',
  version: '3.1',
  status: 'recovered',
  priority: 6,
  weight: 0.5807,
  score: 0.4132,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_037',
  name: 'node_037',
  version: '5.7',
  status: 'pending',
  priority: 3,
  weight: 0.1127,
  score: 0.8015,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_038',
  name: 'node_038',
  version: '3.6',
  status: 'stable',
  priority: 5,
  weight: 0.9704,
  score: 0.0561,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_02_state_handlers_1_039',
  name: 'node_039',
  version: '5.8',
  status: 'degraded',
  priority: 8,
  weight: 0.3236,
  score: 0.9035,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: false
});
