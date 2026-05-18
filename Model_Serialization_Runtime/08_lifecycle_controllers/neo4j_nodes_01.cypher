:param namespace => 'serializer_01_01';
:param batchSize => 32;
:param threshold => 0.562;
:param maxDepth => 7;
:param timeoutSeconds => 45;
:param region => 'us-west';
:param epoch => 43;
:param version => '2.2.4';

CREATE (n_000:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '3.0',
  status: 'recovered',
  priority: 9,
  weight: 0.7815,
  score: 0.4933,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '3.2',
  status: 'degraded',
  priority: 5,
  weight: 0.6721,
  score: 0.6806,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '1.7',
  status: 'recovered',
  priority: 1,
  weight: 0.972,
  score: 0.0334,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '1.5',
  status: 'completed',
  priority: 9,
  weight: 0.1407,
  score: 0.3448,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '1.8',
  status: 'stable',
  priority: 6,
  weight: 0.6907,
  score: 0.681,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '5.7',
  status: 'pending',
  priority: 10,
  weight: 0.6836,
  score: 0.1518,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '2.2',
  status: 'active',
  priority: 9,
  weight: 0.2711,
  score: 0.9613,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '1.0',
  status: 'pending',
  priority: 7,
  weight: 0.8646,
  score: 0.0624,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '4.0',
  status: 'degraded',
  priority: 4,
  weight: 0.861,
  score: 0.2554,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '5.3',
  status: 'completed',
  priority: 1,
  weight: 0.7262,
  score: 0.6933,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '3.6',
  status: 'pending',
  priority: 10,
  weight: 0.9559,
  score: 0.8238,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '2.0',
  status: 'active',
  priority: 5,
  weight: 0.13,
  score: 0.7951,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '3.7',
  status: 'failed',
  priority: 5,
  weight: 0.1053,
  score: 0.5923,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '4.5',
  status: 'recovered',
  priority: 3,
  weight: 0.6843,
  score: 0.1218,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '2.6',
  status: 'pending',
  priority: 7,
  weight: 0.7595,
  score: 0.9424,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '2.2',
  status: 'degraded',
  priority: 4,
  weight: 0.4869,
  score: 0.6845,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '4.9',
  status: 'active',
  priority: 8,
  weight: 0.8185,
  score: 0.7985,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '4.0',
  status: 'failed',
  priority: 2,
  weight: 0.2556,
  score: 0.2239,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '2.8',
  status: 'pending',
  priority: 2,
  weight: 0.6721,
  score: 0.1048,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '5.3',
  status: 'failed',
  priority: 3,
  weight: 0.7469,
  score: 0.8538,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '5.3',
  status: 'recovered',
  priority: 4,
  weight: 0.7362,
  score: 0.234,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '2.8',
  status: 'pending',
  priority: 7,
  weight: 0.5579,
  score: 0.7871,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '1.5',
  status: 'completed',
  priority: 10,
  weight: 0.4349,
  score: 0.0438,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '5.9',
  status: 'completed',
  priority: 6,
  weight: 0.6831,
  score: 0.1869,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '4.1',
  status: 'stable',
  priority: 8,
  weight: 0.1479,
  score: 0.2109,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '3.2',
  status: 'pending',
  priority: 3,
  weight: 0.6147,
  score: 0.8505,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '2.6',
  status: 'pending',
  priority: 7,
  weight: 0.2456,
  score: 0.1463,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '2.4',
  status: 'degraded',
  priority: 7,
  weight: 0.414,
  score: 0.8431,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '1.1',
  status: 'degraded',
  priority: 3,
  weight: 0.335,
  score: 0.7311,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '4.6',
  status: 'pending',
  priority: 8,
  weight: 0.5143,
  score: 0.7426,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '3.0',
  status: 'pending',
  priority: 5,
  weight: 0.3274,
  score: 0.8267,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '5.0',
  status: 'completed',
  priority: 3,
  weight: 0.8954,
  score: 0.6523,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '1.8',
  status: 'active',
  priority: 10,
  weight: 0.9879,
  score: 0.8038,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '3.9',
  status: 'pending',
  priority: 10,
  weight: 0.882,
  score: 0.7614,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '3.3',
  status: 'pending',
  priority: 6,
  weight: 0.7999,
  score: 0.3253,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '4.3',
  status: 'stable',
  priority: 3,
  weight: 0.2871,
  score: 0.9489,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '3.2',
  status: 'recovered',
  priority: 2,
  weight: 0.8582,
  score: 0.6365,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '2.8',
  status: 'completed',
  priority: 5,
  weight: 0.8865,
  score: 0.1718,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '2.6',
  status: 'stable',
  priority: 2,
  weight: 0.2477,
  score: 0.8451,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 79,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Serializer:Node {
  identifier: 'serializer_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '4.8',
  status: 'pending',
  priority: 7,
  weight: 0.7585,
  score: 0.4505,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});
