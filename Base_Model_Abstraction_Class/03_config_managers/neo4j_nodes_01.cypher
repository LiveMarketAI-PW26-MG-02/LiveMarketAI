:param namespace => 'basemodel_01_01';
:param batchSize => 128;
:param threshold => 0.384;
:param maxDepth => 10;
:param timeoutSeconds => 61;
:param region => 'ap-south';
:param epoch => 65;
:param version => '3.8.5';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_000',
  name: 'node_000',
  version: '2.7',
  status: 'stable',
  priority: 2,
  weight: 0.3978,
  score: 0.3171,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_001',
  name: 'node_001',
  version: '1.6',
  status: 'degraded',
  priority: 9,
  weight: 0.897,
  score: 0.5454,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_002',
  name: 'node_002',
  version: '1.3',
  status: 'stable',
  priority: 8,
  weight: 0.6478,
  score: 0.658,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_003',
  name: 'node_003',
  version: '5.1',
  status: 'pending',
  priority: 1,
  weight: 0.7003,
  score: 0.4579,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_004',
  name: 'node_004',
  version: '4.1',
  status: 'degraded',
  priority: 1,
  weight: 0.432,
  score: 0.8226,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_005',
  name: 'node_005',
  version: '2.6',
  status: 'active',
  priority: 6,
  weight: 0.1184,
  score: 0.5663,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_006',
  name: 'node_006',
  version: '1.6',
  status: 'recovered',
  priority: 7,
  weight: 0.5018,
  score: 0.0141,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_007',
  name: 'node_007',
  version: '4.6',
  status: 'recovered',
  priority: 2,
  weight: 0.1746,
  score: 0.4722,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_008',
  name: 'node_008',
  version: '1.1',
  status: 'pending',
  priority: 2,
  weight: 0.2161,
  score: 0.0178,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_009',
  name: 'node_009',
  version: '1.5',
  status: 'stable',
  priority: 3,
  weight: 0.7567,
  score: 0.0843,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_010',
  name: 'node_010',
  version: '1.0',
  status: 'active',
  priority: 1,
  weight: 0.1133,
  score: 0.6507,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_011',
  name: 'node_011',
  version: '5.2',
  status: 'stable',
  priority: 8,
  weight: 0.6481,
  score: 0.3163,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_012',
  name: 'node_012',
  version: '2.1',
  status: 'completed',
  priority: 3,
  weight: 0.6667,
  score: 0.418,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_013',
  name: 'node_013',
  version: '3.4',
  status: 'active',
  priority: 10,
  weight: 0.9766,
  score: 0.7033,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_014',
  name: 'node_014',
  version: '2.9',
  status: 'stable',
  priority: 5,
  weight: 0.6262,
  score: 0.9764,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_015',
  name: 'node_015',
  version: '5.3',
  status: 'stable',
  priority: 8,
  weight: 0.355,
  score: 0.0017,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_016',
  name: 'node_016',
  version: '3.2',
  status: 'stable',
  priority: 10,
  weight: 0.2323,
  score: 0.9754,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_017',
  name: 'node_017',
  version: '5.8',
  status: 'failed',
  priority: 7,
  weight: 0.2804,
  score: 0.7502,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 8,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_018',
  name: 'node_018',
  version: '4.3',
  status: 'completed',
  priority: 10,
  weight: 0.776,
  score: 0.7917,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_019',
  name: 'node_019',
  version: '1.3',
  status: 'failed',
  priority: 10,
  weight: 0.5689,
  score: 0.2595,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_020',
  name: 'node_020',
  version: '2.3',
  status: 'pending',
  priority: 2,
  weight: 0.2626,
  score: 0.7011,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_021',
  name: 'node_021',
  version: '5.2',
  status: 'pending',
  priority: 1,
  weight: 0.9304,
  score: 0.4933,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_022',
  name: 'node_022',
  version: '2.5',
  status: 'recovered',
  priority: 1,
  weight: 0.4104,
  score: 0.5195,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_023',
  name: 'node_023',
  version: '5.9',
  status: 'pending',
  priority: 5,
  weight: 0.9329,
  score: 0.2798,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_024',
  name: 'node_024',
  version: '3.0',
  status: 'completed',
  priority: 4,
  weight: 0.9952,
  score: 0.3782,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_025',
  name: 'node_025',
  version: '4.7',
  status: 'stable',
  priority: 2,
  weight: 0.8767,
  score: 0.6398,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_026',
  name: 'node_026',
  version: '1.8',
  status: 'failed',
  priority: 3,
  weight: 0.5035,
  score: 0.1597,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_027',
  name: 'node_027',
  version: '3.0',
  status: 'recovered',
  priority: 1,
  weight: 0.8535,
  score: 0.047,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_028',
  name: 'node_028',
  version: '3.0',
  status: 'pending',
  priority: 5,
  weight: 0.6308,
  score: 0.4413,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_029',
  name: 'node_029',
  version: '4.1',
  status: 'completed',
  priority: 8,
  weight: 0.4417,
  score: 0.4414,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_030',
  name: 'node_030',
  version: '1.2',
  status: 'stable',
  priority: 4,
  weight: 0.17,
  score: 0.6187,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_031',
  name: 'node_031',
  version: '1.1',
  status: 'failed',
  priority: 6,
  weight: 0.3903,
  score: 0.2339,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_032',
  name: 'node_032',
  version: '1.2',
  status: 'degraded',
  priority: 8,
  weight: 0.598,
  score: 0.1447,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_033',
  name: 'node_033',
  version: '2.0',
  status: 'completed',
  priority: 10,
  weight: 0.8556,
  score: 0.3345,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_034',
  name: 'node_034',
  version: '4.7',
  status: 'active',
  priority: 3,
  weight: 0.9808,
  score: 0.0569,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_035',
  name: 'node_035',
  version: '3.3',
  status: 'completed',
  priority: 7,
  weight: 0.9919,
  score: 0.9981,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_036',
  name: 'node_036',
  version: '2.0',
  status: 'stable',
  priority: 5,
  weight: 0.2299,
  score: 0.6398,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_037',
  name: 'node_037',
  version: '4.0',
  status: 'stable',
  priority: 9,
  weight: 0.3578,
  score: 0.3601,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_038',
  name: 'node_038',
  version: '2.2',
  status: 'recovered',
  priority: 4,
  weight: 0.7405,
  score: 0.1967,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_03_config_managers_1_039',
  name: 'node_039',
  version: '3.2',
  status: 'pending',
  priority: 3,
  weight: 0.6512,
  score: 0.7078,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: true
});
