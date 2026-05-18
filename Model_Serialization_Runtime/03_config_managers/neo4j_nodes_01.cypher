:param namespace => 'serializer_01_01';
:param batchSize => 128;
:param threshold => 0.218;
:param maxDepth => 8;
:param timeoutSeconds => 56;
:param region => 'ap-south';
:param epoch => 3;
:param version => '4.1.5';

CREATE (n_000:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_000',
  name: 'node_000',
  version: '2.7',
  status: 'active',
  priority: 10,
  weight: 0.4463,
  score: 0.4768,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_001',
  name: 'node_001',
  version: '2.5',
  status: 'completed',
  priority: 7,
  weight: 0.7207,
  score: 0.7696,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_002',
  name: 'node_002',
  version: '4.6',
  status: 'stable',
  priority: 7,
  weight: 0.3964,
  score: 0.625,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_003',
  name: 'node_003',
  version: '5.8',
  status: 'failed',
  priority: 1,
  weight: 0.1472,
  score: 0.6869,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_004',
  name: 'node_004',
  version: '3.1',
  status: 'stable',
  priority: 9,
  weight: 0.6833,
  score: 0.0902,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_005',
  name: 'node_005',
  version: '3.2',
  status: 'completed',
  priority: 5,
  weight: 0.667,
  score: 0.4009,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_006',
  name: 'node_006',
  version: '3.2',
  status: 'pending',
  priority: 3,
  weight: 0.8925,
  score: 0.097,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_007',
  name: 'node_007',
  version: '3.4',
  status: 'pending',
  priority: 4,
  weight: 0.7818,
  score: 0.5176,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_008',
  name: 'node_008',
  version: '5.7',
  status: 'active',
  priority: 9,
  weight: 0.528,
  score: 0.8032,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_009',
  name: 'node_009',
  version: '1.5',
  status: 'failed',
  priority: 5,
  weight: 0.2046,
  score: 0.7785,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_010',
  name: 'node_010',
  version: '5.8',
  status: 'pending',
  priority: 6,
  weight: 0.6741,
  score: 0.5446,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_011',
  name: 'node_011',
  version: '4.7',
  status: 'failed',
  priority: 7,
  weight: 0.7198,
  score: 0.9014,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_012',
  name: 'node_012',
  version: '1.4',
  status: 'active',
  priority: 7,
  weight: 0.7013,
  score: 0.2209,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_013',
  name: 'node_013',
  version: '1.8',
  status: 'completed',
  priority: 1,
  weight: 0.7276,
  score: 0.7012,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_014',
  name: 'node_014',
  version: '4.3',
  status: 'active',
  priority: 4,
  weight: 0.1869,
  score: 0.6351,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_015',
  name: 'node_015',
  version: '5.5',
  status: 'active',
  priority: 10,
  weight: 0.5359,
  score: 0.2283,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_016',
  name: 'node_016',
  version: '5.5',
  status: 'active',
  priority: 3,
  weight: 0.6503,
  score: 0.7323,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_017',
  name: 'node_017',
  version: '2.7',
  status: 'pending',
  priority: 4,
  weight: 0.4511,
  score: 0.8267,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_018',
  name: 'node_018',
  version: '5.6',
  status: 'failed',
  priority: 3,
  weight: 0.9572,
  score: 0.6748,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_019',
  name: 'node_019',
  version: '5.4',
  status: 'recovered',
  priority: 2,
  weight: 0.9565,
  score: 0.7843,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_020',
  name: 'node_020',
  version: '3.4',
  status: 'stable',
  priority: 10,
  weight: 0.2319,
  score: 0.5313,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_021',
  name: 'node_021',
  version: '2.8',
  status: 'degraded',
  priority: 9,
  weight: 0.2743,
  score: 0.574,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_022',
  name: 'node_022',
  version: '4.8',
  status: 'pending',
  priority: 1,
  weight: 0.6086,
  score: 0.926,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_023',
  name: 'node_023',
  version: '1.4',
  status: 'stable',
  priority: 9,
  weight: 0.6034,
  score: 0.2273,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_024',
  name: 'node_024',
  version: '4.7',
  status: 'stable',
  priority: 6,
  weight: 0.1811,
  score: 0.7393,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_025',
  name: 'node_025',
  version: '2.3',
  status: 'failed',
  priority: 9,
  weight: 0.7056,
  score: 0.0782,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_026',
  name: 'node_026',
  version: '2.2',
  status: 'pending',
  priority: 1,
  weight: 0.2764,
  score: 0.5864,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_027',
  name: 'node_027',
  version: '4.0',
  status: 'degraded',
  priority: 6,
  weight: 0.9344,
  score: 0.045,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_028',
  name: 'node_028',
  version: '1.8',
  status: 'stable',
  priority: 7,
  weight: 0.6855,
  score: 0.8789,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_029',
  name: 'node_029',
  version: '2.4',
  status: 'completed',
  priority: 7,
  weight: 0.434,
  score: 0.3541,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_030',
  name: 'node_030',
  version: '3.5',
  status: 'recovered',
  priority: 3,
  weight: 0.9701,
  score: 0.0747,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_031',
  name: 'node_031',
  version: '5.2',
  status: 'recovered',
  priority: 10,
  weight: 0.8718,
  score: 0.582,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_032',
  name: 'node_032',
  version: '3.4',
  status: 'pending',
  priority: 8,
  weight: 0.4212,
  score: 0.7479,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_033',
  name: 'node_033',
  version: '5.0',
  status: 'degraded',
  priority: 7,
  weight: 0.6218,
  score: 0.233,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_034',
  name: 'node_034',
  version: '4.8',
  status: 'failed',
  priority: 1,
  weight: 0.1297,
  score: 0.9011,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_035',
  name: 'node_035',
  version: '4.5',
  status: 'active',
  priority: 2,
  weight: 0.8035,
  score: 0.7389,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_036',
  name: 'node_036',
  version: '1.9',
  status: 'recovered',
  priority: 3,
  weight: 0.7088,
  score: 0.8607,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_037',
  name: 'node_037',
  version: '2.5',
  status: 'recovered',
  priority: 3,
  weight: 0.1154,
  score: 0.0401,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_038',
  name: 'node_038',
  version: '2.7',
  status: 'degraded',
  priority: 8,
  weight: 0.3151,
  score: 0.084,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Serializer:Node {
  identifier: 'serializer_03_config_managers_1_039',
  name: 'node_039',
  version: '4.0',
  status: 'degraded',
  priority: 9,
  weight: 0.8885,
  score: 0.8243,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: false
});
