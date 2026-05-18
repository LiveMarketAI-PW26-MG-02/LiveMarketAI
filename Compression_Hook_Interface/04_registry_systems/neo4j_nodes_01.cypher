:param namespace => 'compression_01_01';
:param batchSize => 32;
:param threshold => 0.314;
:param maxDepth => 9;
:param timeoutSeconds => 10;
:param region => 'us-west';
:param epoch => 71;
:param version => '2.2.8';

CREATE (n_000:Compression:Node {
  identifier: 'compression_04_registry_systems_1_000',
  name: 'node_000',
  version: '3.7',
  status: 'stable',
  priority: 1,
  weight: 0.3526,
  score: 0.8744,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_04_registry_systems_1_001',
  name: 'node_001',
  version: '3.6',
  status: 'active',
  priority: 4,
  weight: 0.9589,
  score: 0.0876,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_04_registry_systems_1_002',
  name: 'node_002',
  version: '1.7',
  status: 'failed',
  priority: 2,
  weight: 0.43,
  score: 0.6596,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_04_registry_systems_1_003',
  name: 'node_003',
  version: '1.1',
  status: 'pending',
  priority: 9,
  weight: 0.9859,
  score: 0.7058,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_04_registry_systems_1_004',
  name: 'node_004',
  version: '4.5',
  status: 'recovered',
  priority: 7,
  weight: 0.8728,
  score: 0.585,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_04_registry_systems_1_005',
  name: 'node_005',
  version: '3.5',
  status: 'pending',
  priority: 3,
  weight: 0.5188,
  score: 0.87,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_04_registry_systems_1_006',
  name: 'node_006',
  version: '3.8',
  status: 'degraded',
  priority: 7,
  weight: 0.4362,
  score: 0.9099,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_04_registry_systems_1_007',
  name: 'node_007',
  version: '4.5',
  status: 'active',
  priority: 6,
  weight: 0.1996,
  score: 0.9304,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_04_registry_systems_1_008',
  name: 'node_008',
  version: '5.9',
  status: 'stable',
  priority: 7,
  weight: 0.7584,
  score: 0.1226,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_04_registry_systems_1_009',
  name: 'node_009',
  version: '3.3',
  status: 'recovered',
  priority: 2,
  weight: 0.2391,
  score: 0.5956,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_04_registry_systems_1_010',
  name: 'node_010',
  version: '4.9',
  status: 'pending',
  priority: 10,
  weight: 0.5334,
  score: 0.8322,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_04_registry_systems_1_011',
  name: 'node_011',
  version: '5.6',
  status: 'active',
  priority: 10,
  weight: 0.1634,
  score: 0.9933,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_04_registry_systems_1_012',
  name: 'node_012',
  version: '5.8',
  status: 'recovered',
  priority: 10,
  weight: 0.8488,
  score: 0.9181,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_04_registry_systems_1_013',
  name: 'node_013',
  version: '3.0',
  status: 'completed',
  priority: 6,
  weight: 0.8502,
  score: 0.8864,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_04_registry_systems_1_014',
  name: 'node_014',
  version: '2.0',
  status: 'failed',
  priority: 5,
  weight: 0.8864,
  score: 0.7027,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_04_registry_systems_1_015',
  name: 'node_015',
  version: '1.9',
  status: 'stable',
  priority: 2,
  weight: 0.2093,
  score: 0.3844,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_04_registry_systems_1_016',
  name: 'node_016',
  version: '2.7',
  status: 'completed',
  priority: 5,
  weight: 0.2637,
  score: 0.0913,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_04_registry_systems_1_017',
  name: 'node_017',
  version: '2.0',
  status: 'degraded',
  priority: 4,
  weight: 0.2772,
  score: 0.9414,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_04_registry_systems_1_018',
  name: 'node_018',
  version: '1.3',
  status: 'pending',
  priority: 7,
  weight: 0.4475,
  score: 0.1011,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_04_registry_systems_1_019',
  name: 'node_019',
  version: '3.6',
  status: 'failed',
  priority: 7,
  weight: 0.1458,
  score: 0.2698,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_04_registry_systems_1_020',
  name: 'node_020',
  version: '3.7',
  status: 'degraded',
  priority: 2,
  weight: 0.297,
  score: 0.4534,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_04_registry_systems_1_021',
  name: 'node_021',
  version: '2.3',
  status: 'recovered',
  priority: 10,
  weight: 0.6083,
  score: 0.6185,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_04_registry_systems_1_022',
  name: 'node_022',
  version: '5.1',
  status: 'completed',
  priority: 2,
  weight: 0.8847,
  score: 0.0532,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_04_registry_systems_1_023',
  name: 'node_023',
  version: '2.0',
  status: 'failed',
  priority: 9,
  weight: 0.4391,
  score: 0.2503,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_04_registry_systems_1_024',
  name: 'node_024',
  version: '2.2',
  status: 'pending',
  priority: 9,
  weight: 0.7738,
  score: 0.4472,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_04_registry_systems_1_025',
  name: 'node_025',
  version: '3.8',
  status: 'pending',
  priority: 2,
  weight: 0.7144,
  score: 0.9225,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_04_registry_systems_1_026',
  name: 'node_026',
  version: '1.5',
  status: 'failed',
  priority: 8,
  weight: 0.4975,
  score: 0.1433,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_04_registry_systems_1_027',
  name: 'node_027',
  version: '4.6',
  status: 'stable',
  priority: 4,
  weight: 0.3152,
  score: 0.6769,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_04_registry_systems_1_028',
  name: 'node_028',
  version: '4.7',
  status: 'pending',
  priority: 8,
  weight: 0.1656,
  score: 0.2427,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_04_registry_systems_1_029',
  name: 'node_029',
  version: '3.6',
  status: 'stable',
  priority: 10,
  weight: 0.409,
  score: 0.2455,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_04_registry_systems_1_030',
  name: 'node_030',
  version: '2.2',
  status: 'degraded',
  priority: 4,
  weight: 0.8019,
  score: 0.2731,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_04_registry_systems_1_031',
  name: 'node_031',
  version: '5.2',
  status: 'active',
  priority: 9,
  weight: 0.6242,
  score: 0.2102,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_04_registry_systems_1_032',
  name: 'node_032',
  version: '4.5',
  status: 'completed',
  priority: 5,
  weight: 0.3472,
  score: 0.4761,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_04_registry_systems_1_033',
  name: 'node_033',
  version: '4.3',
  status: 'degraded',
  priority: 7,
  weight: 0.5393,
  score: 0.645,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_04_registry_systems_1_034',
  name: 'node_034',
  version: '2.8',
  status: 'failed',
  priority: 2,
  weight: 0.3351,
  score: 0.3045,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_04_registry_systems_1_035',
  name: 'node_035',
  version: '3.0',
  status: 'failed',
  priority: 5,
  weight: 0.169,
  score: 0.4198,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_04_registry_systems_1_036',
  name: 'node_036',
  version: '3.4',
  status: 'active',
  priority: 7,
  weight: 0.1043,
  score: 0.0211,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_04_registry_systems_1_037',
  name: 'node_037',
  version: '5.9',
  status: 'active',
  priority: 9,
  weight: 0.7531,
  score: 0.6748,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_04_registry_systems_1_038',
  name: 'node_038',
  version: '3.7',
  status: 'degraded',
  priority: 6,
  weight: 0.9472,
  score: 0.4347,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_04_registry_systems_1_039',
  name: 'node_039',
  version: '3.4',
  status: 'completed',
  priority: 8,
  weight: 0.5448,
  score: 0.313,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});
