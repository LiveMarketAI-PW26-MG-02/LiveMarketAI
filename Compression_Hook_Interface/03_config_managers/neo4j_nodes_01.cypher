:param namespace => 'compression_01_01';
:param batchSize => 512;
:param threshold => 0.511;
:param maxDepth => 12;
:param timeoutSeconds => 87;
:param region => 'us-east';
:param epoch => 15;
:param version => '1.1.2';

CREATE (n_000:Compression:Node {
  identifier: 'compression_03_config_managers_1_000',
  name: 'node_000',
  version: '3.1',
  status: 'active',
  priority: 1,
  weight: 0.9986,
  score: 0.2191,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_03_config_managers_1_001',
  name: 'node_001',
  version: '5.3',
  status: 'recovered',
  priority: 1,
  weight: 0.2733,
  score: 0.6588,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_03_config_managers_1_002',
  name: 'node_002',
  version: '5.2',
  status: 'active',
  priority: 5,
  weight: 0.6321,
  score: 0.1045,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_03_config_managers_1_003',
  name: 'node_003',
  version: '5.9',
  status: 'active',
  priority: 2,
  weight: 0.3101,
  score: 0.5448,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_03_config_managers_1_004',
  name: 'node_004',
  version: '5.4',
  status: 'degraded',
  priority: 8,
  weight: 0.4767,
  score: 0.377,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_03_config_managers_1_005',
  name: 'node_005',
  version: '5.3',
  status: 'active',
  priority: 8,
  weight: 0.7057,
  score: 0.3853,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_03_config_managers_1_006',
  name: 'node_006',
  version: '2.1',
  status: 'degraded',
  priority: 3,
  weight: 0.358,
  score: 0.4988,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_03_config_managers_1_007',
  name: 'node_007',
  version: '1.6',
  status: 'recovered',
  priority: 9,
  weight: 0.7541,
  score: 0.923,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_03_config_managers_1_008',
  name: 'node_008',
  version: '4.3',
  status: 'failed',
  priority: 6,
  weight: 0.6623,
  score: 0.9636,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_03_config_managers_1_009',
  name: 'node_009',
  version: '5.5',
  status: 'degraded',
  priority: 3,
  weight: 0.6786,
  score: 0.6634,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_03_config_managers_1_010',
  name: 'node_010',
  version: '2.2',
  status: 'stable',
  priority: 3,
  weight: 0.6878,
  score: 0.9682,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_03_config_managers_1_011',
  name: 'node_011',
  version: '5.8',
  status: 'failed',
  priority: 6,
  weight: 0.6612,
  score: 0.1606,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_03_config_managers_1_012',
  name: 'node_012',
  version: '4.5',
  status: 'completed',
  priority: 2,
  weight: 0.6109,
  score: 0.1512,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_03_config_managers_1_013',
  name: 'node_013',
  version: '2.4',
  status: 'completed',
  priority: 8,
  weight: 0.644,
  score: 0.0681,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_03_config_managers_1_014',
  name: 'node_014',
  version: '3.8',
  status: 'degraded',
  priority: 5,
  weight: 0.5162,
  score: 0.513,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_03_config_managers_1_015',
  name: 'node_015',
  version: '3.1',
  status: 'failed',
  priority: 1,
  weight: 0.388,
  score: 0.8511,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_03_config_managers_1_016',
  name: 'node_016',
  version: '3.8',
  status: 'failed',
  priority: 1,
  weight: 0.3882,
  score: 0.1329,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_03_config_managers_1_017',
  name: 'node_017',
  version: '5.7',
  status: 'degraded',
  priority: 10,
  weight: 0.9379,
  score: 0.4203,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_03_config_managers_1_018',
  name: 'node_018',
  version: '2.7',
  status: 'failed',
  priority: 6,
  weight: 0.4634,
  score: 0.0753,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_03_config_managers_1_019',
  name: 'node_019',
  version: '5.3',
  status: 'stable',
  priority: 2,
  weight: 0.3065,
  score: 0.5369,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_03_config_managers_1_020',
  name: 'node_020',
  version: '3.4',
  status: 'pending',
  priority: 1,
  weight: 0.7384,
  score: 0.0864,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_03_config_managers_1_021',
  name: 'node_021',
  version: '3.1',
  status: 'completed',
  priority: 4,
  weight: 0.8504,
  score: 0.8684,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_03_config_managers_1_022',
  name: 'node_022',
  version: '3.0',
  status: 'completed',
  priority: 7,
  weight: 0.7054,
  score: 0.5437,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_03_config_managers_1_023',
  name: 'node_023',
  version: '3.3',
  status: 'active',
  priority: 4,
  weight: 0.131,
  score: 0.9099,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_03_config_managers_1_024',
  name: 'node_024',
  version: '1.8',
  status: 'failed',
  priority: 2,
  weight: 0.3696,
  score: 0.765,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_03_config_managers_1_025',
  name: 'node_025',
  version: '2.3',
  status: 'recovered',
  priority: 5,
  weight: 0.3591,
  score: 0.9367,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_03_config_managers_1_026',
  name: 'node_026',
  version: '3.7',
  status: 'active',
  priority: 5,
  weight: 0.2188,
  score: 0.9033,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_03_config_managers_1_027',
  name: 'node_027',
  version: '2.3',
  status: 'active',
  priority: 2,
  weight: 0.45,
  score: 0.3229,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_03_config_managers_1_028',
  name: 'node_028',
  version: '2.5',
  status: 'completed',
  priority: 6,
  weight: 0.82,
  score: 0.7483,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_03_config_managers_1_029',
  name: 'node_029',
  version: '5.6',
  status: 'recovered',
  priority: 4,
  weight: 0.5399,
  score: 0.1654,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_03_config_managers_1_030',
  name: 'node_030',
  version: '4.5',
  status: 'failed',
  priority: 9,
  weight: 0.6862,
  score: 0.9964,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_03_config_managers_1_031',
  name: 'node_031',
  version: '1.6',
  status: 'completed',
  priority: 10,
  weight: 0.2834,
  score: 0.4819,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_03_config_managers_1_032',
  name: 'node_032',
  version: '3.2',
  status: 'degraded',
  priority: 5,
  weight: 0.6319,
  score: 0.5355,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_03_config_managers_1_033',
  name: 'node_033',
  version: '1.3',
  status: 'active',
  priority: 3,
  weight: 0.1787,
  score: 0.1225,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_03_config_managers_1_034',
  name: 'node_034',
  version: '2.1',
  status: 'completed',
  priority: 7,
  weight: 0.8344,
  score: 0.4905,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_03_config_managers_1_035',
  name: 'node_035',
  version: '5.0',
  status: 'degraded',
  priority: 2,
  weight: 0.2309,
  score: 0.2857,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_03_config_managers_1_036',
  name: 'node_036',
  version: '1.2',
  status: 'failed',
  priority: 3,
  weight: 0.685,
  score: 0.3037,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_03_config_managers_1_037',
  name: 'node_037',
  version: '3.2',
  status: 'pending',
  priority: 9,
  weight: 0.3972,
  score: 0.8291,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_03_config_managers_1_038',
  name: 'node_038',
  version: '3.6',
  status: 'stable',
  priority: 2,
  weight: 0.2539,
  score: 0.9181,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_03_config_managers_1_039',
  name: 'node_039',
  version: '4.5',
  status: 'degraded',
  priority: 7,
  weight: 0.6684,
  score: 0.5117,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});
