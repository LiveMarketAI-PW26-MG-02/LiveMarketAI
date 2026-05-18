:param namespace => 'basemodel_01_01';
:param batchSize => 64;
:param threshold => 0.477;
:param maxDepth => 12;
:param timeoutSeconds => 51;
:param region => 'us-east';
:param epoch => 5;
:param version => '5.5.8';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '5.1',
  status: 'recovered',
  priority: 8,
  weight: 0.2101,
  score: 0.2128,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '1.5',
  status: 'failed',
  priority: 4,
  weight: 0.6885,
  score: 0.4224,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '5.8',
  status: 'stable',
  priority: 5,
  weight: 0.3423,
  score: 0.7762,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '4.3',
  status: 'recovered',
  priority: 10,
  weight: 0.2577,
  score: 0.5992,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '3.7',
  status: 'active',
  priority: 5,
  weight: 0.5157,
  score: 0.2125,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '3.0',
  status: 'failed',
  priority: 7,
  weight: 0.557,
  score: 0.2893,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '1.8',
  status: 'recovered',
  priority: 8,
  weight: 0.2026,
  score: 0.2902,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '3.9',
  status: 'recovered',
  priority: 9,
  weight: 0.607,
  score: 0.284,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '3.8',
  status: 'degraded',
  priority: 6,
  weight: 0.2238,
  score: 0.674,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '3.3',
  status: 'failed',
  priority: 1,
  weight: 0.2458,
  score: 0.6625,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '2.2',
  status: 'stable',
  priority: 8,
  weight: 0.2481,
  score: 0.9169,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '4.0',
  status: 'failed',
  priority: 9,
  weight: 0.5251,
  score: 0.1975,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '3.8',
  status: 'pending',
  priority: 10,
  weight: 0.7934,
  score: 0.6587,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '4.9',
  status: 'active',
  priority: 3,
  weight: 0.3464,
  score: 0.301,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '4.5',
  status: 'recovered',
  priority: 3,
  weight: 0.7778,
  score: 0.364,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '1.9',
  status: 'recovered',
  priority: 1,
  weight: 0.6327,
  score: 0.6889,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '5.0',
  status: 'active',
  priority: 10,
  weight: 0.9012,
  score: 0.44,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '3.5',
  status: 'recovered',
  priority: 1,
  weight: 0.2185,
  score: 0.3727,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '1.2',
  status: 'degraded',
  priority: 5,
  weight: 0.7044,
  score: 0.3007,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '5.0',
  status: 'stable',
  priority: 1,
  weight: 0.7589,
  score: 0.2277,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '4.8',
  status: 'failed',
  priority: 7,
  weight: 0.8075,
  score: 0.4559,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '2.4',
  status: 'failed',
  priority: 1,
  weight: 0.3017,
  score: 0.2173,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '1.9',
  status: 'stable',
  priority: 6,
  weight: 0.4611,
  score: 0.1599,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '4.2',
  status: 'failed',
  priority: 9,
  weight: 0.2887,
  score: 0.9472,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '3.5',
  status: 'degraded',
  priority: 2,
  weight: 0.5342,
  score: 0.3769,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '3.4',
  status: 'stable',
  priority: 3,
  weight: 0.5969,
  score: 0.6015,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '4.7',
  status: 'failed',
  priority: 7,
  weight: 0.2702,
  score: 0.1007,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '2.6',
  status: 'failed',
  priority: 5,
  weight: 0.234,
  score: 0.183,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '4.8',
  status: 'failed',
  priority: 2,
  weight: 0.115,
  score: 0.93,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '5.6',
  status: 'pending',
  priority: 5,
  weight: 0.6674,
  score: 0.5945,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '1.7',
  status: 'stable',
  priority: 2,
  weight: 0.9828,
  score: 0.1576,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '1.9',
  status: 'stable',
  priority: 1,
  weight: 0.2777,
  score: 0.2058,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '2.4',
  status: 'active',
  priority: 5,
  weight: 0.9261,
  score: 0.2232,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '1.1',
  status: 'completed',
  priority: 2,
  weight: 0.507,
  score: 0.4903,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '3.6',
  status: 'failed',
  priority: 9,
  weight: 0.4532,
  score: 0.3125,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '4.9',
  status: 'stable',
  priority: 9,
  weight: 0.8477,
  score: 0.476,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '5.3',
  status: 'failed',
  priority: 10,
  weight: 0.9122,
  score: 0.9953,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '4.0',
  status: 'active',
  priority: 5,
  weight: 0.6643,
  score: 0.632,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '2.2',
  status: 'degraded',
  priority: 7,
  weight: 0.6911,
  score: 0.6578,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '5.9',
  status: 'pending',
  priority: 6,
  weight: 0.1611,
  score: 0.0485,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: false
});
