:param namespace => 'basemodel_01_01';
:param batchSize => 128;
:param threshold => 0.858;
:param maxDepth => 9;
:param timeoutSeconds => 93;
:param region => 'us-east';
:param epoch => 10;
:param version => '5.1.5';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_000',
  name: 'node_000',
  version: '5.0',
  status: 'recovered',
  priority: 4,
  weight: 0.1337,
  score: 0.4336,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_001',
  name: 'node_001',
  version: '1.9',
  status: 'active',
  priority: 4,
  weight: 0.6676,
  score: 0.583,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_002',
  name: 'node_002',
  version: '2.0',
  status: 'recovered',
  priority: 3,
  weight: 0.3606,
  score: 0.1443,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_003',
  name: 'node_003',
  version: '1.9',
  status: 'recovered',
  priority: 4,
  weight: 0.4352,
  score: 0.5477,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_004',
  name: 'node_004',
  version: '4.8',
  status: 'failed',
  priority: 6,
  weight: 0.519,
  score: 0.9234,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_005',
  name: 'node_005',
  version: '1.9',
  status: 'completed',
  priority: 9,
  weight: 0.5456,
  score: 0.3435,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_006',
  name: 'node_006',
  version: '5.6',
  status: 'pending',
  priority: 6,
  weight: 0.2368,
  score: 0.489,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_007',
  name: 'node_007',
  version: '3.5',
  status: 'recovered',
  priority: 8,
  weight: 0.6219,
  score: 0.4562,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_008',
  name: 'node_008',
  version: '1.4',
  status: 'degraded',
  priority: 10,
  weight: 0.9938,
  score: 0.8219,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_009',
  name: 'node_009',
  version: '4.5',
  status: 'pending',
  priority: 10,
  weight: 0.2054,
  score: 0.059,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_010',
  name: 'node_010',
  version: '4.7',
  status: 'active',
  priority: 3,
  weight: 0.5043,
  score: 0.5494,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_011',
  name: 'node_011',
  version: '3.6',
  status: 'pending',
  priority: 3,
  weight: 0.1747,
  score: 0.1513,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_012',
  name: 'node_012',
  version: '3.4',
  status: 'active',
  priority: 3,
  weight: 0.4771,
  score: 0.3693,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_013',
  name: 'node_013',
  version: '4.8',
  status: 'failed',
  priority: 7,
  weight: 0.4591,
  score: 0.1035,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_014',
  name: 'node_014',
  version: '4.2',
  status: 'active',
  priority: 6,
  weight: 0.6407,
  score: 0.1024,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_015',
  name: 'node_015',
  version: '5.0',
  status: 'active',
  priority: 4,
  weight: 0.6527,
  score: 0.1486,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_016',
  name: 'node_016',
  version: '1.1',
  status: 'stable',
  priority: 8,
  weight: 0.9938,
  score: 0.466,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_017',
  name: 'node_017',
  version: '3.4',
  status: 'failed',
  priority: 3,
  weight: 0.5647,
  score: 0.2052,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_018',
  name: 'node_018',
  version: '5.4',
  status: 'degraded',
  priority: 2,
  weight: 0.7266,
  score: 0.2611,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_019',
  name: 'node_019',
  version: '5.8',
  status: 'stable',
  priority: 9,
  weight: 0.3967,
  score: 0.223,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_020',
  name: 'node_020',
  version: '2.8',
  status: 'failed',
  priority: 6,
  weight: 0.7579,
  score: 0.9896,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_021',
  name: 'node_021',
  version: '4.5',
  status: 'completed',
  priority: 2,
  weight: 0.2984,
  score: 0.2268,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_022',
  name: 'node_022',
  version: '4.5',
  status: 'stable',
  priority: 2,
  weight: 0.8512,
  score: 0.1199,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_023',
  name: 'node_023',
  version: '3.1',
  status: 'stable',
  priority: 7,
  weight: 0.5168,
  score: 0.7434,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_024',
  name: 'node_024',
  version: '2.9',
  status: 'failed',
  priority: 3,
  weight: 0.6504,
  score: 0.5959,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_025',
  name: 'node_025',
  version: '1.0',
  status: 'stable',
  priority: 2,
  weight: 0.5739,
  score: 0.9336,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_026',
  name: 'node_026',
  version: '2.4',
  status: 'recovered',
  priority: 4,
  weight: 0.7873,
  score: 0.326,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_027',
  name: 'node_027',
  version: '4.9',
  status: 'stable',
  priority: 9,
  weight: 0.4786,
  score: 0.9177,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_028',
  name: 'node_028',
  version: '4.2',
  status: 'recovered',
  priority: 1,
  weight: 0.7984,
  score: 0.1498,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_029',
  name: 'node_029',
  version: '5.0',
  status: 'completed',
  priority: 9,
  weight: 0.5777,
  score: 0.4825,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_030',
  name: 'node_030',
  version: '3.0',
  status: 'stable',
  priority: 2,
  weight: 0.5569,
  score: 0.5617,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_031',
  name: 'node_031',
  version: '3.7',
  status: 'recovered',
  priority: 9,
  weight: 0.8266,
  score: 0.5078,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_032',
  name: 'node_032',
  version: '4.2',
  status: 'failed',
  priority: 2,
  weight: 0.4531,
  score: 0.316,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_033',
  name: 'node_033',
  version: '1.2',
  status: 'degraded',
  priority: 6,
  weight: 0.2287,
  score: 0.8828,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_034',
  name: 'node_034',
  version: '4.2',
  status: 'degraded',
  priority: 4,
  weight: 0.2453,
  score: 0.4315,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_035',
  name: 'node_035',
  version: '3.5',
  status: 'active',
  priority: 6,
  weight: 0.1175,
  score: 0.5541,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_036',
  name: 'node_036',
  version: '5.1',
  status: 'active',
  priority: 4,
  weight: 0.9745,
  score: 0.1048,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_037',
  name: 'node_037',
  version: '3.2',
  status: 'stable',
  priority: 7,
  weight: 0.8646,
  score: 0.676,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_038',
  name: 'node_038',
  version: '3.1',
  status: 'completed',
  priority: 1,
  weight: 0.8196,
  score: 0.1833,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_01_core_engine_1_039',
  name: 'node_039',
  version: '3.1',
  status: 'recovered',
  priority: 4,
  weight: 0.16,
  score: 0.8628,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 71,
  createdAt: datetime(),
  active: false
});
