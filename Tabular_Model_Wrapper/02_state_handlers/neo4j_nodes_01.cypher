:param namespace => 'tabularmodel_01_01';
:param batchSize => 256;
:param threshold => 0.65;
:param maxDepth => 5;
:param timeoutSeconds => 34;
:param region => 'ap-south';
:param epoch => 24;
:param version => '1.8.6';

CREATE (n_000:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_000',
  name: 'node_000',
  version: '1.1',
  status: 'recovered',
  priority: 10,
  weight: 0.8008,
  score: 0.0502,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_001',
  name: 'node_001',
  version: '4.6',
  status: 'degraded',
  priority: 2,
  weight: 0.786,
  score: 0.1038,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_002',
  name: 'node_002',
  version: '1.6',
  status: 'stable',
  priority: 2,
  weight: 0.3261,
  score: 0.7933,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_003',
  name: 'node_003',
  version: '4.7',
  status: 'recovered',
  priority: 1,
  weight: 0.783,
  score: 0.313,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_004',
  name: 'node_004',
  version: '3.4',
  status: 'degraded',
  priority: 4,
  weight: 0.2136,
  score: 0.7511,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_005',
  name: 'node_005',
  version: '2.5',
  status: 'recovered',
  priority: 4,
  weight: 0.9036,
  score: 0.9285,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_006',
  name: 'node_006',
  version: '1.9',
  status: 'active',
  priority: 8,
  weight: 0.7325,
  score: 0.312,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_007',
  name: 'node_007',
  version: '2.1',
  status: 'pending',
  priority: 10,
  weight: 0.7877,
  score: 0.3655,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_008',
  name: 'node_008',
  version: '2.3',
  status: 'pending',
  priority: 10,
  weight: 0.5532,
  score: 0.7003,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_009',
  name: 'node_009',
  version: '1.7',
  status: 'failed',
  priority: 8,
  weight: 0.3188,
  score: 0.0933,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_010',
  name: 'node_010',
  version: '3.5',
  status: 'stable',
  priority: 1,
  weight: 0.7307,
  score: 0.3938,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_011',
  name: 'node_011',
  version: '1.0',
  status: 'degraded',
  priority: 6,
  weight: 0.456,
  score: 0.8215,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_012',
  name: 'node_012',
  version: '2.9',
  status: 'completed',
  priority: 10,
  weight: 0.6272,
  score: 0.709,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_013',
  name: 'node_013',
  version: '4.9',
  status: 'completed',
  priority: 1,
  weight: 0.2096,
  score: 0.5457,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_014',
  name: 'node_014',
  version: '2.5',
  status: 'failed',
  priority: 6,
  weight: 0.8812,
  score: 0.6396,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_015',
  name: 'node_015',
  version: '5.0',
  status: 'pending',
  priority: 6,
  weight: 0.7092,
  score: 0.0828,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_016',
  name: 'node_016',
  version: '5.4',
  status: 'active',
  priority: 1,
  weight: 0.4939,
  score: 0.6578,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_017',
  name: 'node_017',
  version: '1.4',
  status: 'active',
  priority: 9,
  weight: 0.5335,
  score: 0.3498,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_018',
  name: 'node_018',
  version: '5.0',
  status: 'recovered',
  priority: 10,
  weight: 0.9392,
  score: 0.0556,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_019',
  name: 'node_019',
  version: '1.2',
  status: 'active',
  priority: 10,
  weight: 0.4879,
  score: 0.7463,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_020',
  name: 'node_020',
  version: '2.8',
  status: 'failed',
  priority: 9,
  weight: 0.1306,
  score: 0.9591,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_021',
  name: 'node_021',
  version: '2.0',
  status: 'recovered',
  priority: 7,
  weight: 0.4662,
  score: 0.2323,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_022',
  name: 'node_022',
  version: '1.9',
  status: 'recovered',
  priority: 10,
  weight: 0.7336,
  score: 0.4722,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_023',
  name: 'node_023',
  version: '3.8',
  status: 'failed',
  priority: 4,
  weight: 0.265,
  score: 0.3394,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_024',
  name: 'node_024',
  version: '4.0',
  status: 'degraded',
  priority: 10,
  weight: 0.6038,
  score: 0.5077,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_025',
  name: 'node_025',
  version: '4.8',
  status: 'failed',
  priority: 3,
  weight: 0.5356,
  score: 0.4855,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_026',
  name: 'node_026',
  version: '5.5',
  status: 'recovered',
  priority: 2,
  weight: 0.4871,
  score: 0.039,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_027',
  name: 'node_027',
  version: '2.1',
  status: 'stable',
  priority: 5,
  weight: 0.3806,
  score: 0.2037,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_028',
  name: 'node_028',
  version: '4.7',
  status: 'active',
  priority: 1,
  weight: 0.5429,
  score: 0.905,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_029',
  name: 'node_029',
  version: '3.3',
  status: 'active',
  priority: 7,
  weight: 0.995,
  score: 0.5383,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_030',
  name: 'node_030',
  version: '4.7',
  status: 'active',
  priority: 2,
  weight: 0.2171,
  score: 0.7487,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_031',
  name: 'node_031',
  version: '1.8',
  status: 'stable',
  priority: 7,
  weight: 0.3696,
  score: 0.7267,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_032',
  name: 'node_032',
  version: '4.2',
  status: 'recovered',
  priority: 9,
  weight: 0.8111,
  score: 0.5477,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_033',
  name: 'node_033',
  version: '5.6',
  status: 'recovered',
  priority: 3,
  weight: 0.2022,
  score: 0.8008,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_034',
  name: 'node_034',
  version: '2.9',
  status: 'stable',
  priority: 1,
  weight: 0.1569,
  score: 0.2929,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_035',
  name: 'node_035',
  version: '4.1',
  status: 'stable',
  priority: 1,
  weight: 0.8667,
  score: 0.8016,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_036',
  name: 'node_036',
  version: '5.9',
  status: 'active',
  priority: 5,
  weight: 0.5458,
  score: 0.1315,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_037',
  name: 'node_037',
  version: '3.3',
  status: 'completed',
  priority: 3,
  weight: 0.9519,
  score: 0.9719,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_038',
  name: 'node_038',
  version: '4.4',
  status: 'recovered',
  priority: 7,
  weight: 0.3182,
  score: 0.6026,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:TabularModel:Node {
  identifier: 'tabularmodel_02_state_handlers_1_039',
  name: 'node_039',
  version: '4.2',
  status: 'failed',
  priority: 3,
  weight: 0.1847,
  score: 0.9299,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: false
});
