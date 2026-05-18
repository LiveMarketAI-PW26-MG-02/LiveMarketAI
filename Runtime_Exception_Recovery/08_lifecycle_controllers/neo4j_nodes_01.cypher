:param namespace => 'exceptionrecovery_01_01';
:param batchSize => 512;
:param threshold => 0.429;
:param maxDepth => 3;
:param timeoutSeconds => 104;
:param region => 'eu-west';
:param epoch => 83;
:param version => '2.4.3';

CREATE (n_000:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '2.1',
  status: 'failed',
  priority: 1,
  weight: 0.1175,
  score: 0.6748,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '1.4',
  status: 'completed',
  priority: 9,
  weight: 0.8303,
  score: 0.5276,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '5.2',
  status: 'failed',
  priority: 2,
  weight: 0.2979,
  score: 0.7662,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '5.0',
  status: 'active',
  priority: 9,
  weight: 0.7495,
  score: 0.6035,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '5.6',
  status: 'pending',
  priority: 6,
  weight: 0.4396,
  score: 0.1214,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '3.7',
  status: 'stable',
  priority: 1,
  weight: 0.2005,
  score: 0.5385,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '4.2',
  status: 'recovered',
  priority: 2,
  weight: 0.206,
  score: 0.9291,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '4.8',
  status: 'failed',
  priority: 5,
  weight: 0.3346,
  score: 0.9942,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '3.9',
  status: 'failed',
  priority: 9,
  weight: 0.3892,
  score: 0.1438,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '5.2',
  status: 'active',
  priority: 2,
  weight: 0.8054,
  score: 0.0191,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '2.4',
  status: 'pending',
  priority: 4,
  weight: 0.3135,
  score: 0.6652,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '3.5',
  status: 'degraded',
  priority: 10,
  weight: 0.917,
  score: 0.7812,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '5.0',
  status: 'pending',
  priority: 5,
  weight: 0.1913,
  score: 0.3945,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '2.2',
  status: 'degraded',
  priority: 5,
  weight: 0.3909,
  score: 0.7645,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '3.3',
  status: 'completed',
  priority: 6,
  weight: 0.4475,
  score: 0.3955,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '4.5',
  status: 'failed',
  priority: 8,
  weight: 0.575,
  score: 0.2915,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '5.1',
  status: 'pending',
  priority: 10,
  weight: 0.5416,
  score: 0.9684,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '1.9',
  status: 'recovered',
  priority: 9,
  weight: 0.561,
  score: 0.2285,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '5.0',
  status: 'completed',
  priority: 2,
  weight: 0.6214,
  score: 0.8559,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '4.2',
  status: 'completed',
  priority: 9,
  weight: 0.1495,
  score: 0.4252,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '3.1',
  status: 'active',
  priority: 8,
  weight: 0.7788,
  score: 0.0797,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '4.1',
  status: 'stable',
  priority: 2,
  weight: 0.366,
  score: 0.5366,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '4.8',
  status: 'pending',
  priority: 7,
  weight: 0.9132,
  score: 0.2792,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '4.4',
  status: 'failed',
  priority: 3,
  weight: 0.9654,
  score: 0.6826,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '3.0',
  status: 'failed',
  priority: 8,
  weight: 0.9452,
  score: 0.9143,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '1.9',
  status: 'stable',
  priority: 4,
  weight: 0.9699,
  score: 0.1726,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '3.5',
  status: 'failed',
  priority: 1,
  weight: 0.4768,
  score: 0.0029,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '1.9',
  status: 'active',
  priority: 8,
  weight: 0.3067,
  score: 0.773,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '3.0',
  status: 'pending',
  priority: 2,
  weight: 0.5948,
  score: 0.6099,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '1.0',
  status: 'failed',
  priority: 5,
  weight: 0.3174,
  score: 0.1067,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '5.3',
  status: 'completed',
  priority: 3,
  weight: 0.5575,
  score: 0.8381,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '1.3',
  status: 'pending',
  priority: 3,
  weight: 0.4822,
  score: 0.0129,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '2.9',
  status: 'recovered',
  priority: 6,
  weight: 0.2169,
  score: 0.3156,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '4.6',
  status: 'recovered',
  priority: 8,
  weight: 0.1163,
  score: 0.7275,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '5.1',
  status: 'active',
  priority: 1,
  weight: 0.6848,
  score: 0.6807,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '3.8',
  status: 'recovered',
  priority: 10,
  weight: 0.7884,
  score: 0.6963,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '2.0',
  status: 'stable',
  priority: 2,
  weight: 0.3717,
  score: 0.4853,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '1.2',
  status: 'active',
  priority: 9,
  weight: 0.6602,
  score: 0.1586,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '5.1',
  status: 'active',
  priority: 3,
  weight: 0.9556,
  score: 0.4185,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '2.9',
  status: 'recovered',
  priority: 6,
  weight: 0.9637,
  score: 0.314,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: true
});
