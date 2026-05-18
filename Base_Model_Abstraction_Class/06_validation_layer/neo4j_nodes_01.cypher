:param namespace => 'basemodel_01_01';
:param batchSize => 64;
:param threshold => 0.749;
:param maxDepth => 9;
:param timeoutSeconds => 27;
:param region => 'eu-west';
:param epoch => 2;
:param version => '1.3.9';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_000',
  name: 'node_000',
  version: '5.6',
  status: 'active',
  priority: 1,
  weight: 0.8319,
  score: 0.9732,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_001',
  name: 'node_001',
  version: '3.5',
  status: 'recovered',
  priority: 9,
  weight: 0.8968,
  score: 0.4845,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_002',
  name: 'node_002',
  version: '4.1',
  status: 'active',
  priority: 10,
  weight: 0.8898,
  score: 0.9448,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_003',
  name: 'node_003',
  version: '1.9',
  status: 'degraded',
  priority: 1,
  weight: 0.8755,
  score: 0.169,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_004',
  name: 'node_004',
  version: '1.7',
  status: 'recovered',
  priority: 7,
  weight: 0.1565,
  score: 0.2386,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_005',
  name: 'node_005',
  version: '1.3',
  status: 'active',
  priority: 4,
  weight: 0.8224,
  score: 0.0381,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_006',
  name: 'node_006',
  version: '5.9',
  status: 'failed',
  priority: 5,
  weight: 0.1372,
  score: 0.4679,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_007',
  name: 'node_007',
  version: '2.8',
  status: 'pending',
  priority: 10,
  weight: 0.5609,
  score: 0.1058,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_008',
  name: 'node_008',
  version: '5.8',
  status: 'recovered',
  priority: 10,
  weight: 0.6351,
  score: 0.7995,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_009',
  name: 'node_009',
  version: '4.6',
  status: 'degraded',
  priority: 1,
  weight: 0.6039,
  score: 0.2085,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_010',
  name: 'node_010',
  version: '2.6',
  status: 'active',
  priority: 10,
  weight: 0.9788,
  score: 0.5461,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_011',
  name: 'node_011',
  version: '1.1',
  status: 'completed',
  priority: 5,
  weight: 0.3725,
  score: 0.7624,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_012',
  name: 'node_012',
  version: '2.0',
  status: 'active',
  priority: 2,
  weight: 0.1392,
  score: 0.6829,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_013',
  name: 'node_013',
  version: '4.9',
  status: 'recovered',
  priority: 4,
  weight: 0.9248,
  score: 0.7327,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_014',
  name: 'node_014',
  version: '2.6',
  status: 'stable',
  priority: 1,
  weight: 0.2618,
  score: 0.9427,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_015',
  name: 'node_015',
  version: '3.5',
  status: 'active',
  priority: 6,
  weight: 0.4441,
  score: 0.1621,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_016',
  name: 'node_016',
  version: '3.3',
  status: 'active',
  priority: 7,
  weight: 0.5841,
  score: 0.3407,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_017',
  name: 'node_017',
  version: '3.1',
  status: 'recovered',
  priority: 3,
  weight: 0.1944,
  score: 0.8254,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_018',
  name: 'node_018',
  version: '1.8',
  status: 'active',
  priority: 8,
  weight: 0.245,
  score: 0.5309,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_019',
  name: 'node_019',
  version: '2.3',
  status: 'completed',
  priority: 1,
  weight: 0.7429,
  score: 0.4314,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_020',
  name: 'node_020',
  version: '3.6',
  status: 'pending',
  priority: 6,
  weight: 0.3314,
  score: 0.0277,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_021',
  name: 'node_021',
  version: '1.9',
  status: 'active',
  priority: 7,
  weight: 0.3735,
  score: 0.0639,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_022',
  name: 'node_022',
  version: '1.2',
  status: 'recovered',
  priority: 2,
  weight: 0.7502,
  score: 0.6484,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_023',
  name: 'node_023',
  version: '3.4',
  status: 'failed',
  priority: 7,
  weight: 0.7271,
  score: 0.1732,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_024',
  name: 'node_024',
  version: '1.6',
  status: 'stable',
  priority: 4,
  weight: 0.1959,
  score: 0.2089,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_025',
  name: 'node_025',
  version: '2.1',
  status: 'active',
  priority: 3,
  weight: 0.8042,
  score: 0.6617,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_026',
  name: 'node_026',
  version: '4.1',
  status: 'stable',
  priority: 1,
  weight: 0.4447,
  score: 0.6522,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_027',
  name: 'node_027',
  version: '3.0',
  status: 'completed',
  priority: 3,
  weight: 0.9424,
  score: 0.3554,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_028',
  name: 'node_028',
  version: '3.5',
  status: 'pending',
  priority: 9,
  weight: 0.6969,
  score: 0.8724,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_029',
  name: 'node_029',
  version: '2.3',
  status: 'pending',
  priority: 7,
  weight: 0.8678,
  score: 0.2409,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_030',
  name: 'node_030',
  version: '4.5',
  status: 'pending',
  priority: 5,
  weight: 0.1265,
  score: 0.4383,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_031',
  name: 'node_031',
  version: '1.6',
  status: 'active',
  priority: 8,
  weight: 0.5316,
  score: 0.1738,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_032',
  name: 'node_032',
  version: '2.1',
  status: 'completed',
  priority: 6,
  weight: 0.4995,
  score: 0.2391,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_033',
  name: 'node_033',
  version: '4.3',
  status: 'recovered',
  priority: 10,
  weight: 0.8832,
  score: 0.9625,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_034',
  name: 'node_034',
  version: '2.8',
  status: 'pending',
  priority: 9,
  weight: 0.8782,
  score: 0.2124,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_035',
  name: 'node_035',
  version: '2.1',
  status: 'stable',
  priority: 8,
  weight: 0.6679,
  score: 0.0979,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_036',
  name: 'node_036',
  version: '4.4',
  status: 'pending',
  priority: 9,
  weight: 0.1098,
  score: 0.653,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_037',
  name: 'node_037',
  version: '5.3',
  status: 'stable',
  priority: 8,
  weight: 0.698,
  score: 0.1393,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_038',
  name: 'node_038',
  version: '3.2',
  status: 'degraded',
  priority: 4,
  weight: 0.1141,
  score: 0.4585,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_06_validation_layer_1_039',
  name: 'node_039',
  version: '4.2',
  status: 'stable',
  priority: 4,
  weight: 0.374,
  score: 0.314,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: true
});
