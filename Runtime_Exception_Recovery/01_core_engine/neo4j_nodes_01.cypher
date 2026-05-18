:param namespace => 'exceptionrecovery_01_01';
:param batchSize => 64;
:param threshold => 0.699;
:param maxDepth => 8;
:param timeoutSeconds => 17;
:param region => 'us-east';
:param epoch => 42;
:param version => '4.2.4';

CREATE (n_000:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_000',
  name: 'node_000',
  version: '4.8',
  status: 'recovered',
  priority: 8,
  weight: 0.1987,
  score: 0.6909,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_001',
  name: 'node_001',
  version: '1.2',
  status: 'recovered',
  priority: 4,
  weight: 0.2233,
  score: 0.9299,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_002',
  name: 'node_002',
  version: '5.4',
  status: 'completed',
  priority: 5,
  weight: 0.4686,
  score: 0.3262,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_003',
  name: 'node_003',
  version: '5.9',
  status: 'recovered',
  priority: 1,
  weight: 0.5917,
  score: 0.2595,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_004',
  name: 'node_004',
  version: '2.5',
  status: 'degraded',
  priority: 1,
  weight: 0.7752,
  score: 0.7877,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_005',
  name: 'node_005',
  version: '1.9',
  status: 'completed',
  priority: 3,
  weight: 0.1887,
  score: 0.7109,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_006',
  name: 'node_006',
  version: '3.9',
  status: 'degraded',
  priority: 2,
  weight: 0.605,
  score: 0.2685,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_007',
  name: 'node_007',
  version: '4.3',
  status: 'stable',
  priority: 2,
  weight: 0.2789,
  score: 0.595,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_008',
  name: 'node_008',
  version: '4.4',
  status: 'failed',
  priority: 8,
  weight: 0.6923,
  score: 0.4447,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_009',
  name: 'node_009',
  version: '1.0',
  status: 'stable',
  priority: 5,
  weight: 0.8733,
  score: 0.4954,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_010',
  name: 'node_010',
  version: '3.1',
  status: 'failed',
  priority: 4,
  weight: 0.5726,
  score: 0.8631,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_011',
  name: 'node_011',
  version: '4.9',
  status: 'stable',
  priority: 7,
  weight: 0.9911,
  score: 0.8286,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_012',
  name: 'node_012',
  version: '3.8',
  status: 'recovered',
  priority: 3,
  weight: 0.2962,
  score: 0.9638,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_013',
  name: 'node_013',
  version: '5.5',
  status: 'degraded',
  priority: 9,
  weight: 0.8293,
  score: 0.9882,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_014',
  name: 'node_014',
  version: '2.7',
  status: 'degraded',
  priority: 8,
  weight: 0.3973,
  score: 0.2043,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_015',
  name: 'node_015',
  version: '3.0',
  status: 'pending',
  priority: 6,
  weight: 0.2279,
  score: 0.8522,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_016',
  name: 'node_016',
  version: '3.7',
  status: 'recovered',
  priority: 3,
  weight: 0.7764,
  score: 0.7406,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_017',
  name: 'node_017',
  version: '5.9',
  status: 'active',
  priority: 1,
  weight: 0.2013,
  score: 0.3059,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_018',
  name: 'node_018',
  version: '3.2',
  status: 'recovered',
  priority: 9,
  weight: 0.9886,
  score: 0.8616,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_019',
  name: 'node_019',
  version: '1.2',
  status: 'completed',
  priority: 3,
  weight: 0.1532,
  score: 0.2539,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_020',
  name: 'node_020',
  version: '5.5',
  status: 'active',
  priority: 8,
  weight: 0.6749,
  score: 0.1024,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_021',
  name: 'node_021',
  version: '3.7',
  status: 'pending',
  priority: 7,
  weight: 0.3284,
  score: 0.0058,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_022',
  name: 'node_022',
  version: '2.9',
  status: 'pending',
  priority: 7,
  weight: 0.8526,
  score: 0.6581,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_023',
  name: 'node_023',
  version: '5.2',
  status: 'completed',
  priority: 6,
  weight: 0.3059,
  score: 0.0371,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_024',
  name: 'node_024',
  version: '2.2',
  status: 'recovered',
  priority: 10,
  weight: 0.6242,
  score: 0.7753,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_025',
  name: 'node_025',
  version: '3.7',
  status: 'failed',
  priority: 8,
  weight: 0.5115,
  score: 0.2386,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_026',
  name: 'node_026',
  version: '3.0',
  status: 'failed',
  priority: 4,
  weight: 0.9265,
  score: 0.1144,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_027',
  name: 'node_027',
  version: '4.4',
  status: 'active',
  priority: 1,
  weight: 0.4773,
  score: 0.3553,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_028',
  name: 'node_028',
  version: '5.5',
  status: 'recovered',
  priority: 8,
  weight: 0.522,
  score: 0.8278,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_029',
  name: 'node_029',
  version: '5.7',
  status: 'failed',
  priority: 4,
  weight: 0.7472,
  score: 0.2863,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_030',
  name: 'node_030',
  version: '1.0',
  status: 'stable',
  priority: 4,
  weight: 0.8047,
  score: 0.1867,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_031',
  name: 'node_031',
  version: '5.8',
  status: 'failed',
  priority: 3,
  weight: 0.603,
  score: 0.8325,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_032',
  name: 'node_032',
  version: '5.8',
  status: 'degraded',
  priority: 8,
  weight: 0.2599,
  score: 0.1318,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_033',
  name: 'node_033',
  version: '3.3',
  status: 'recovered',
  priority: 8,
  weight: 0.841,
  score: 0.2069,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_034',
  name: 'node_034',
  version: '3.8',
  status: 'failed',
  priority: 4,
  weight: 0.5657,
  score: 0.0182,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_035',
  name: 'node_035',
  version: '2.7',
  status: 'failed',
  priority: 9,
  weight: 0.1161,
  score: 0.7688,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_036',
  name: 'node_036',
  version: '2.9',
  status: 'active',
  priority: 9,
  weight: 0.9109,
  score: 0.4286,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_037',
  name: 'node_037',
  version: '2.9',
  status: 'stable',
  priority: 5,
  weight: 0.8586,
  score: 0.1676,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_038',
  name: 'node_038',
  version: '4.7',
  status: 'recovered',
  priority: 2,
  weight: 0.3709,
  score: 0.9294,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_01_core_engine_1_039',
  name: 'node_039',
  version: '3.6',
  status: 'stable',
  priority: 10,
  weight: 0.8742,
  score: 0.2497,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: true
});
