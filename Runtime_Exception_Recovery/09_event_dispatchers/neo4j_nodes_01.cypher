:param namespace => 'exceptionrecovery_01_01';
:param batchSize => 128;
:param threshold => 0.115;
:param maxDepth => 10;
:param timeoutSeconds => 59;
:param region => 'eu-west';
:param epoch => 83;
:param version => '5.4.3';

CREATE (n_000:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_000',
  name: 'node_000',
  version: '1.1',
  status: 'degraded',
  priority: 5,
  weight: 0.3398,
  score: 0.7498,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_001',
  name: 'node_001',
  version: '1.0',
  status: 'degraded',
  priority: 10,
  weight: 0.2985,
  score: 0.6222,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_002',
  name: 'node_002',
  version: '2.9',
  status: 'stable',
  priority: 7,
  weight: 0.8543,
  score: 0.7378,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_003',
  name: 'node_003',
  version: '4.8',
  status: 'stable',
  priority: 9,
  weight: 0.1709,
  score: 0.1137,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_004',
  name: 'node_004',
  version: '5.8',
  status: 'recovered',
  priority: 2,
  weight: 0.5412,
  score: 0.1884,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_005',
  name: 'node_005',
  version: '2.6',
  status: 'stable',
  priority: 8,
  weight: 0.8277,
  score: 0.4909,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_006',
  name: 'node_006',
  version: '2.2',
  status: 'completed',
  priority: 5,
  weight: 0.5829,
  score: 0.7188,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_007',
  name: 'node_007',
  version: '4.6',
  status: 'completed',
  priority: 7,
  weight: 0.2708,
  score: 0.9583,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_008',
  name: 'node_008',
  version: '2.7',
  status: 'failed',
  priority: 3,
  weight: 0.18,
  score: 0.9662,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_009',
  name: 'node_009',
  version: '4.0',
  status: 'stable',
  priority: 9,
  weight: 0.2508,
  score: 0.7493,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_010',
  name: 'node_010',
  version: '5.2',
  status: 'stable',
  priority: 2,
  weight: 0.2519,
  score: 0.5065,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_011',
  name: 'node_011',
  version: '1.4',
  status: 'stable',
  priority: 2,
  weight: 0.5222,
  score: 0.1817,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_012',
  name: 'node_012',
  version: '3.9',
  status: 'completed',
  priority: 2,
  weight: 0.8689,
  score: 0.8527,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_013',
  name: 'node_013',
  version: '2.8',
  status: 'completed',
  priority: 4,
  weight: 0.5268,
  score: 0.1025,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_014',
  name: 'node_014',
  version: '4.5',
  status: 'recovered',
  priority: 5,
  weight: 0.4426,
  score: 0.0798,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_015',
  name: 'node_015',
  version: '2.7',
  status: 'stable',
  priority: 10,
  weight: 0.1087,
  score: 0.2341,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_016',
  name: 'node_016',
  version: '3.5',
  status: 'recovered',
  priority: 2,
  weight: 0.3893,
  score: 0.381,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_017',
  name: 'node_017',
  version: '2.3',
  status: 'pending',
  priority: 8,
  weight: 0.83,
  score: 0.413,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_018',
  name: 'node_018',
  version: '1.6',
  status: 'failed',
  priority: 3,
  weight: 0.1808,
  score: 0.0668,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_019',
  name: 'node_019',
  version: '3.1',
  status: 'completed',
  priority: 8,
  weight: 0.1519,
  score: 0.7549,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_020',
  name: 'node_020',
  version: '1.2',
  status: 'stable',
  priority: 1,
  weight: 0.6717,
  score: 0.1165,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_021',
  name: 'node_021',
  version: '2.6',
  status: 'pending',
  priority: 4,
  weight: 0.1948,
  score: 0.8755,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_022',
  name: 'node_022',
  version: '2.9',
  status: 'completed',
  priority: 3,
  weight: 0.2854,
  score: 0.1019,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_023',
  name: 'node_023',
  version: '1.3',
  status: 'recovered',
  priority: 7,
  weight: 0.6083,
  score: 0.4253,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_024',
  name: 'node_024',
  version: '4.1',
  status: 'recovered',
  priority: 2,
  weight: 0.5432,
  score: 0.2653,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_025',
  name: 'node_025',
  version: '4.6',
  status: 'failed',
  priority: 6,
  weight: 0.6783,
  score: 0.812,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_026',
  name: 'node_026',
  version: '4.7',
  status: 'failed',
  priority: 7,
  weight: 0.3853,
  score: 0.4324,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_027',
  name: 'node_027',
  version: '5.2',
  status: 'active',
  priority: 1,
  weight: 0.7193,
  score: 0.4224,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_028',
  name: 'node_028',
  version: '5.3',
  status: 'active',
  priority: 7,
  weight: 0.7269,
  score: 0.1568,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_029',
  name: 'node_029',
  version: '5.1',
  status: 'recovered',
  priority: 6,
  weight: 0.8683,
  score: 0.4108,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_030',
  name: 'node_030',
  version: '3.4',
  status: 'recovered',
  priority: 1,
  weight: 0.5422,
  score: 0.073,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_031',
  name: 'node_031',
  version: '5.7',
  status: 'degraded',
  priority: 3,
  weight: 0.9655,
  score: 0.9974,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_032',
  name: 'node_032',
  version: '4.6',
  status: 'stable',
  priority: 1,
  weight: 0.2014,
  score: 0.0854,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_033',
  name: 'node_033',
  version: '5.6',
  status: 'degraded',
  priority: 5,
  weight: 0.7673,
  score: 0.1061,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_034',
  name: 'node_034',
  version: '5.0',
  status: 'active',
  priority: 3,
  weight: 0.5611,
  score: 0.8332,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_035',
  name: 'node_035',
  version: '2.6',
  status: 'recovered',
  priority: 9,
  weight: 0.1382,
  score: 0.4768,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_036',
  name: 'node_036',
  version: '3.7',
  status: 'completed',
  priority: 1,
  weight: 0.2062,
  score: 0.5511,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_037',
  name: 'node_037',
  version: '1.5',
  status: 'recovered',
  priority: 4,
  weight: 0.3952,
  score: 0.9377,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_038',
  name: 'node_038',
  version: '2.8',
  status: 'failed',
  priority: 1,
  weight: 0.2716,
  score: 0.8162,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_09_event_dispatchers_1_039',
  name: 'node_039',
  version: '5.1',
  status: 'failed',
  priority: 10,
  weight: 0.6539,
  score: 0.9476,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});
