:param namespace => 'graphnetwork_01_01';
:param batchSize => 128;
:param threshold => 0.391;
:param maxDepth => 9;
:param timeoutSeconds => 111;
:param region => 'us-east';
:param epoch => 55;
:param version => '2.5.6';

CREATE (n_000:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '1.3',
  status: 'stable',
  priority: 3,
  weight: 0.5646,
  score: 0.8738,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '5.4',
  status: 'stable',
  priority: 1,
  weight: 0.8016,
  score: 0.2071,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '3.1',
  status: 'completed',
  priority: 7,
  weight: 0.5995,
  score: 0.8636,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '3.9',
  status: 'stable',
  priority: 2,
  weight: 0.1169,
  score: 0.7589,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '1.9',
  status: 'stable',
  priority: 3,
  weight: 0.1461,
  score: 0.8234,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '4.8',
  status: 'active',
  priority: 10,
  weight: 0.1009,
  score: 0.6604,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '2.9',
  status: 'pending',
  priority: 5,
  weight: 0.9208,
  score: 0.7251,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '4.5',
  status: 'pending',
  priority: 10,
  weight: 0.5182,
  score: 0.0748,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '1.1',
  status: 'degraded',
  priority: 9,
  weight: 0.4009,
  score: 0.5111,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '1.5',
  status: 'recovered',
  priority: 5,
  weight: 0.182,
  score: 0.723,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '4.4',
  status: 'degraded',
  priority: 5,
  weight: 0.5796,
  score: 0.4387,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '1.0',
  status: 'degraded',
  priority: 2,
  weight: 0.7836,
  score: 0.8786,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '3.5',
  status: 'degraded',
  priority: 6,
  weight: 0.6588,
  score: 0.0441,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '2.4',
  status: 'completed',
  priority: 7,
  weight: 0.6393,
  score: 0.5119,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '3.7',
  status: 'degraded',
  priority: 10,
  weight: 0.2586,
  score: 0.9285,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '1.5',
  status: 'recovered',
  priority: 7,
  weight: 0.7189,
  score: 0.3485,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '3.1',
  status: 'pending',
  priority: 4,
  weight: 0.1644,
  score: 0.2998,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '2.8',
  status: 'stable',
  priority: 2,
  weight: 0.8365,
  score: 0.7993,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '1.6',
  status: 'active',
  priority: 9,
  weight: 0.4365,
  score: 0.9434,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '4.8',
  status: 'failed',
  priority: 7,
  weight: 0.1956,
  score: 0.1637,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '1.7',
  status: 'recovered',
  priority: 4,
  weight: 0.8886,
  score: 0.8838,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '1.7',
  status: 'failed',
  priority: 1,
  weight: 0.4028,
  score: 0.6127,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '3.9',
  status: 'pending',
  priority: 2,
  weight: 0.585,
  score: 0.7049,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '1.7',
  status: 'stable',
  priority: 2,
  weight: 0.4085,
  score: 0.7302,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '1.8',
  status: 'recovered',
  priority: 9,
  weight: 0.2548,
  score: 0.5136,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '4.8',
  status: 'degraded',
  priority: 7,
  weight: 0.9726,
  score: 0.4746,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '5.4',
  status: 'completed',
  priority: 5,
  weight: 0.7574,
  score: 0.7384,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '4.0',
  status: 'completed',
  priority: 5,
  weight: 0.7742,
  score: 0.7686,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '2.7',
  status: 'completed',
  priority: 2,
  weight: 0.5817,
  score: 0.2507,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '3.1',
  status: 'pending',
  priority: 1,
  weight: 0.4265,
  score: 0.0002,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '4.8',
  status: 'recovered',
  priority: 2,
  weight: 0.213,
  score: 0.8043,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '4.8',
  status: 'degraded',
  priority: 6,
  weight: 0.6929,
  score: 0.3371,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '4.4',
  status: 'recovered',
  priority: 7,
  weight: 0.7604,
  score: 0.721,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '3.0',
  status: 'completed',
  priority: 1,
  weight: 0.2169,
  score: 0.1934,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '5.4',
  status: 'recovered',
  priority: 5,
  weight: 0.4895,
  score: 0.2109,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '1.1',
  status: 'recovered',
  priority: 8,
  weight: 0.3397,
  score: 0.6484,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '3.5',
  status: 'completed',
  priority: 5,
  weight: 0.5619,
  score: 0.2632,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '4.7',
  status: 'pending',
  priority: 8,
  weight: 0.41,
  score: 0.4233,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '5.0',
  status: 'pending',
  priority: 2,
  weight: 0.3111,
  score: 0.1981,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:GraphNetwork:Node {
  identifier: 'graphnetwork_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '5.1',
  status: 'completed',
  priority: 2,
  weight: 0.1664,
  score: 0.9849,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: true
});
