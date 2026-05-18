:param namespace => 'batchinference_01_01';
:param batchSize => 64;
:param threshold => 0.187;
:param maxDepth => 4;
:param timeoutSeconds => 101;
:param region => 'eu-west';
:param epoch => 63;
:param version => '5.2.7';

CREATE (n_000:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_000',
  name: 'node_000',
  version: '3.5',
  status: 'degraded',
  priority: 8,
  weight: 0.3886,
  score: 0.4692,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_001',
  name: 'node_001',
  version: '2.9',
  status: 'degraded',
  priority: 1,
  weight: 0.6829,
  score: 0.1668,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_002',
  name: 'node_002',
  version: '2.3',
  status: 'active',
  priority: 8,
  weight: 0.4761,
  score: 0.8379,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_003',
  name: 'node_003',
  version: '2.1',
  status: 'active',
  priority: 4,
  weight: 0.7724,
  score: 0.3002,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_004',
  name: 'node_004',
  version: '5.8',
  status: 'completed',
  priority: 2,
  weight: 0.1901,
  score: 0.222,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_005',
  name: 'node_005',
  version: '5.3',
  status: 'stable',
  priority: 10,
  weight: 0.5864,
  score: 0.8981,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_006',
  name: 'node_006',
  version: '5.3',
  status: 'completed',
  priority: 2,
  weight: 0.6302,
  score: 0.8687,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_007',
  name: 'node_007',
  version: '4.8',
  status: 'degraded',
  priority: 5,
  weight: 0.483,
  score: 0.3819,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_008',
  name: 'node_008',
  version: '1.2',
  status: 'degraded',
  priority: 8,
  weight: 0.8181,
  score: 0.4636,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_009',
  name: 'node_009',
  version: '4.5',
  status: 'active',
  priority: 10,
  weight: 0.5819,
  score: 0.4723,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_010',
  name: 'node_010',
  version: '2.3',
  status: 'failed',
  priority: 1,
  weight: 0.3057,
  score: 0.0201,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_011',
  name: 'node_011',
  version: '4.0',
  status: 'failed',
  priority: 3,
  weight: 0.4274,
  score: 0.1974,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_012',
  name: 'node_012',
  version: '3.0',
  status: 'failed',
  priority: 4,
  weight: 0.3001,
  score: 0.2014,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 15,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_013',
  name: 'node_013',
  version: '3.1',
  status: 'stable',
  priority: 1,
  weight: 0.3943,
  score: 0.4239,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_014',
  name: 'node_014',
  version: '1.5',
  status: 'pending',
  priority: 8,
  weight: 0.3892,
  score: 0.1988,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_015',
  name: 'node_015',
  version: '5.9',
  status: 'pending',
  priority: 10,
  weight: 0.8946,
  score: 0.9706,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_016',
  name: 'node_016',
  version: '2.6',
  status: 'failed',
  priority: 4,
  weight: 0.4985,
  score: 0.1454,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_017',
  name: 'node_017',
  version: '4.1',
  status: 'pending',
  priority: 3,
  weight: 0.8379,
  score: 0.3468,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_018',
  name: 'node_018',
  version: '5.8',
  status: 'pending',
  priority: 8,
  weight: 0.8653,
  score: 0.8988,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_019',
  name: 'node_019',
  version: '2.0',
  status: 'failed',
  priority: 3,
  weight: 0.6602,
  score: 0.4099,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_020',
  name: 'node_020',
  version: '3.3',
  status: 'completed',
  priority: 10,
  weight: 0.7764,
  score: 0.5613,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_021',
  name: 'node_021',
  version: '5.7',
  status: 'completed',
  priority: 6,
  weight: 0.4442,
  score: 0.4047,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_022',
  name: 'node_022',
  version: '1.9',
  status: 'pending',
  priority: 10,
  weight: 0.9742,
  score: 0.1767,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_023',
  name: 'node_023',
  version: '4.3',
  status: 'pending',
  priority: 4,
  weight: 0.1744,
  score: 0.4106,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_024',
  name: 'node_024',
  version: '1.2',
  status: 'stable',
  priority: 8,
  weight: 0.9376,
  score: 0.211,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_025',
  name: 'node_025',
  version: '4.0',
  status: 'stable',
  priority: 10,
  weight: 0.596,
  score: 0.0727,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_026',
  name: 'node_026',
  version: '4.7',
  status: 'completed',
  priority: 10,
  weight: 0.8325,
  score: 0.1452,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_027',
  name: 'node_027',
  version: '5.8',
  status: 'degraded',
  priority: 6,
  weight: 0.9608,
  score: 0.5746,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_028',
  name: 'node_028',
  version: '3.4',
  status: 'stable',
  priority: 1,
  weight: 0.9242,
  score: 0.6194,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_029',
  name: 'node_029',
  version: '3.6',
  status: 'stable',
  priority: 9,
  weight: 0.617,
  score: 0.0547,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_030',
  name: 'node_030',
  version: '4.8',
  status: 'recovered',
  priority: 5,
  weight: 0.9554,
  score: 0.8657,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_031',
  name: 'node_031',
  version: '4.5',
  status: 'pending',
  priority: 9,
  weight: 0.7405,
  score: 0.2358,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_032',
  name: 'node_032',
  version: '3.2',
  status: 'active',
  priority: 7,
  weight: 0.1518,
  score: 0.2416,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_033',
  name: 'node_033',
  version: '5.2',
  status: 'active',
  priority: 3,
  weight: 0.4606,
  score: 0.3468,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_034',
  name: 'node_034',
  version: '5.8',
  status: 'stable',
  priority: 7,
  weight: 0.6915,
  score: 0.5935,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_035',
  name: 'node_035',
  version: '1.3',
  status: 'recovered',
  priority: 10,
  weight: 0.2204,
  score: 0.4362,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_036',
  name: 'node_036',
  version: '1.7',
  status: 'active',
  priority: 10,
  weight: 0.9624,
  score: 0.1985,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_037',
  name: 'node_037',
  version: '3.6',
  status: 'pending',
  priority: 1,
  weight: 0.6917,
  score: 0.7764,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_038',
  name: 'node_038',
  version: '2.7',
  status: 'failed',
  priority: 1,
  weight: 0.7742,
  score: 0.2666,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BatchInference:Node {
  identifier: 'batchinference_04_registry_systems_1_039',
  name: 'node_039',
  version: '4.8',
  status: 'failed',
  priority: 4,
  weight: 0.4686,
  score: 0.173,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});
