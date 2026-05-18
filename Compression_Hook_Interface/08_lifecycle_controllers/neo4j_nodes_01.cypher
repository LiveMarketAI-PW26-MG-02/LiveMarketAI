:param namespace => 'compression_01_01';
:param batchSize => 512;
:param threshold => 0.166;
:param maxDepth => 7;
:param timeoutSeconds => 59;
:param region => 'us-west';
:param epoch => 90;
:param version => '4.4.0';

CREATE (n_000:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '5.3',
  status: 'completed',
  priority: 10,
  weight: 0.2555,
  score: 0.7619,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '5.5',
  status: 'degraded',
  priority: 3,
  weight: 0.9648,
  score: 0.8272,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '5.1',
  status: 'degraded',
  priority: 5,
  weight: 0.9593,
  score: 0.3339,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '3.6',
  status: 'recovered',
  priority: 10,
  weight: 0.5499,
  score: 0.7761,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '1.6',
  status: 'failed',
  priority: 5,
  weight: 0.5923,
  score: 0.5002,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '3.7',
  status: 'completed',
  priority: 10,
  weight: 0.8838,
  score: 0.7316,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '4.6',
  status: 'stable',
  priority: 9,
  weight: 0.6946,
  score: 0.9155,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '2.8',
  status: 'failed',
  priority: 7,
  weight: 0.5849,
  score: 0.0285,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '3.5',
  status: 'pending',
  priority: 7,
  weight: 0.3553,
  score: 0.2265,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '3.7',
  status: 'pending',
  priority: 5,
  weight: 0.8637,
  score: 0.3071,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '3.8',
  status: 'completed',
  priority: 3,
  weight: 0.8025,
  score: 0.7075,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '3.3',
  status: 'active',
  priority: 6,
  weight: 0.8797,
  score: 0.2976,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '5.8',
  status: 'stable',
  priority: 3,
  weight: 0.3081,
  score: 0.906,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '5.1',
  status: 'failed',
  priority: 7,
  weight: 0.4822,
  score: 0.0678,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '2.7',
  status: 'recovered',
  priority: 9,
  weight: 0.5311,
  score: 0.1637,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '3.4',
  status: 'degraded',
  priority: 10,
  weight: 0.3634,
  score: 0.846,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '3.0',
  status: 'completed',
  priority: 2,
  weight: 0.6649,
  score: 0.5065,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '4.5',
  status: 'stable',
  priority: 5,
  weight: 0.2084,
  score: 0.876,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '3.4',
  status: 'completed',
  priority: 5,
  weight: 0.9236,
  score: 0.4555,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '1.9',
  status: 'completed',
  priority: 8,
  weight: 0.4151,
  score: 0.9088,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '5.1',
  status: 'active',
  priority: 10,
  weight: 0.1124,
  score: 0.5593,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '5.2',
  status: 'degraded',
  priority: 6,
  weight: 0.5049,
  score: 0.7694,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '1.7',
  status: 'failed',
  priority: 4,
  weight: 0.3908,
  score: 0.6219,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '1.3',
  status: 'active',
  priority: 3,
  weight: 0.341,
  score: 0.5996,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '3.1',
  status: 'recovered',
  priority: 8,
  weight: 0.62,
  score: 0.2224,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '4.4',
  status: 'active',
  priority: 8,
  weight: 0.3313,
  score: 0.9047,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '2.4',
  status: 'failed',
  priority: 7,
  weight: 0.6268,
  score: 0.4379,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '2.0',
  status: 'degraded',
  priority: 4,
  weight: 0.4671,
  score: 0.742,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '3.5',
  status: 'active',
  priority: 8,
  weight: 0.4064,
  score: 0.849,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '2.0',
  status: 'pending',
  priority: 1,
  weight: 0.8077,
  score: 0.6796,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '4.3',
  status: 'stable',
  priority: 9,
  weight: 0.964,
  score: 0.8847,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '4.3',
  status: 'completed',
  priority: 4,
  weight: 0.3672,
  score: 0.7903,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '4.5',
  status: 'failed',
  priority: 7,
  weight: 0.7584,
  score: 0.1074,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '4.7',
  status: 'active',
  priority: 6,
  weight: 0.8234,
  score: 0.0054,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '2.1',
  status: 'recovered',
  priority: 10,
  weight: 0.7284,
  score: 0.0408,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '3.5',
  status: 'failed',
  priority: 4,
  weight: 0.2071,
  score: 0.3511,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '2.0',
  status: 'active',
  priority: 9,
  weight: 0.2231,
  score: 0.4805,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '2.4',
  status: 'failed',
  priority: 6,
  weight: 0.7385,
  score: 0.5268,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '3.0',
  status: 'recovered',
  priority: 9,
  weight: 0.2629,
  score: 0.5196,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '5.8',
  status: 'recovered',
  priority: 5,
  weight: 0.7557,
  score: 0.2952,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: true
});
