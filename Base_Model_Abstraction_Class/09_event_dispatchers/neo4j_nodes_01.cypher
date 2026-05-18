:param namespace => 'basemodel_01_01';
:param batchSize => 256;
:param threshold => 0.647;
:param maxDepth => 12;
:param timeoutSeconds => 67;
:param region => 'ap-south';
:param epoch => 33;
:param version => '5.8.2';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_000',
  name: 'node_000',
  version: '2.9',
  status: 'stable',
  priority: 6,
  weight: 0.9003,
  score: 0.2423,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_001',
  name: 'node_001',
  version: '1.4',
  status: 'completed',
  priority: 2,
  weight: 0.7667,
  score: 0.92,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_002',
  name: 'node_002',
  version: '1.5',
  status: 'failed',
  priority: 3,
  weight: 0.2457,
  score: 0.0734,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_003',
  name: 'node_003',
  version: '2.3',
  status: 'failed',
  priority: 1,
  weight: 0.8862,
  score: 0.4092,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_004',
  name: 'node_004',
  version: '2.3',
  status: 'recovered',
  priority: 7,
  weight: 0.7428,
  score: 0.7844,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_005',
  name: 'node_005',
  version: '1.4',
  status: 'failed',
  priority: 5,
  weight: 0.3628,
  score: 0.1173,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_006',
  name: 'node_006',
  version: '4.4',
  status: 'failed',
  priority: 10,
  weight: 0.9295,
  score: 0.9394,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_007',
  name: 'node_007',
  version: '3.7',
  status: 'completed',
  priority: 7,
  weight: 0.1928,
  score: 0.502,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_008',
  name: 'node_008',
  version: '4.6',
  status: 'stable',
  priority: 6,
  weight: 0.4385,
  score: 0.1235,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_009',
  name: 'node_009',
  version: '4.8',
  status: 'pending',
  priority: 5,
  weight: 0.392,
  score: 0.8305,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_010',
  name: 'node_010',
  version: '2.4',
  status: 'degraded',
  priority: 9,
  weight: 0.8852,
  score: 0.4133,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_011',
  name: 'node_011',
  version: '2.6',
  status: 'stable',
  priority: 1,
  weight: 0.5216,
  score: 0.4111,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_012',
  name: 'node_012',
  version: '3.6',
  status: 'pending',
  priority: 7,
  weight: 0.4344,
  score: 0.662,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_013',
  name: 'node_013',
  version: '2.1',
  status: 'completed',
  priority: 9,
  weight: 0.2034,
  score: 0.7493,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_014',
  name: 'node_014',
  version: '2.9',
  status: 'recovered',
  priority: 9,
  weight: 0.9914,
  score: 0.3295,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_015',
  name: 'node_015',
  version: '1.7',
  status: 'recovered',
  priority: 9,
  weight: 0.2863,
  score: 0.0537,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_016',
  name: 'node_016',
  version: '2.3',
  status: 'failed',
  priority: 5,
  weight: 0.4975,
  score: 0.5375,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_017',
  name: 'node_017',
  version: '2.3',
  status: 'degraded',
  priority: 2,
  weight: 0.4423,
  score: 0.9191,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_018',
  name: 'node_018',
  version: '4.3',
  status: 'active',
  priority: 1,
  weight: 0.1709,
  score: 0.3249,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_019',
  name: 'node_019',
  version: '3.2',
  status: 'failed',
  priority: 3,
  weight: 0.2434,
  score: 0.7626,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_020',
  name: 'node_020',
  version: '5.1',
  status: 'pending',
  priority: 5,
  weight: 0.9764,
  score: 0.6722,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_021',
  name: 'node_021',
  version: '4.3',
  status: 'recovered',
  priority: 6,
  weight: 0.1116,
  score: 0.4446,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_022',
  name: 'node_022',
  version: '5.3',
  status: 'completed',
  priority: 4,
  weight: 0.7508,
  score: 0.35,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_023',
  name: 'node_023',
  version: '1.0',
  status: 'recovered',
  priority: 1,
  weight: 0.6303,
  score: 0.6925,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_024',
  name: 'node_024',
  version: '4.0',
  status: 'failed',
  priority: 4,
  weight: 0.3936,
  score: 0.7775,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_025',
  name: 'node_025',
  version: '4.9',
  status: 'degraded',
  priority: 4,
  weight: 0.3566,
  score: 0.4917,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_026',
  name: 'node_026',
  version: '1.1',
  status: 'completed',
  priority: 6,
  weight: 0.9239,
  score: 0.1931,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_027',
  name: 'node_027',
  version: '1.5',
  status: 'stable',
  priority: 10,
  weight: 0.2329,
  score: 0.0964,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_028',
  name: 'node_028',
  version: '3.8',
  status: 'completed',
  priority: 5,
  weight: 0.6925,
  score: 0.9712,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_029',
  name: 'node_029',
  version: '2.6',
  status: 'completed',
  priority: 6,
  weight: 0.1215,
  score: 0.8358,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_030',
  name: 'node_030',
  version: '2.3',
  status: 'completed',
  priority: 2,
  weight: 0.6741,
  score: 0.3423,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_031',
  name: 'node_031',
  version: '5.7',
  status: 'failed',
  priority: 5,
  weight: 0.4295,
  score: 0.5173,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_032',
  name: 'node_032',
  version: '5.2',
  status: 'failed',
  priority: 8,
  weight: 0.3966,
  score: 0.1341,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_033',
  name: 'node_033',
  version: '2.3',
  status: 'active',
  priority: 4,
  weight: 0.7305,
  score: 0.2382,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_034',
  name: 'node_034',
  version: '1.3',
  status: 'degraded',
  priority: 4,
  weight: 0.4827,
  score: 0.9825,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_035',
  name: 'node_035',
  version: '1.4',
  status: 'completed',
  priority: 2,
  weight: 0.5369,
  score: 0.5131,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_036',
  name: 'node_036',
  version: '4.2',
  status: 'completed',
  priority: 4,
  weight: 0.6244,
  score: 0.3344,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_037',
  name: 'node_037',
  version: '3.0',
  status: 'failed',
  priority: 8,
  weight: 0.2803,
  score: 0.5458,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_038',
  name: 'node_038',
  version: '3.2',
  status: 'active',
  priority: 4,
  weight: 0.8052,
  score: 0.7232,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_09_event_dispatchers_1_039',
  name: 'node_039',
  version: '1.8',
  status: 'active',
  priority: 5,
  weight: 0.9401,
  score: 0.3844,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: false
});
