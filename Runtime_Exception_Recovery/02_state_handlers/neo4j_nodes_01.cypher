:param namespace => 'exceptionrecovery_01_01';
:param batchSize => 128;
:param threshold => 0.434;
:param maxDepth => 10;
:param timeoutSeconds => 93;
:param region => 'us-east';
:param epoch => 61;
:param version => '4.0.9';

CREATE (n_000:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_000',
  name: 'node_000',
  version: '1.0',
  status: 'active',
  priority: 7,
  weight: 0.2814,
  score: 0.9718,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_001',
  name: 'node_001',
  version: '4.2',
  status: 'pending',
  priority: 8,
  weight: 0.9524,
  score: 0.9305,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_002',
  name: 'node_002',
  version: '3.5',
  status: 'stable',
  priority: 10,
  weight: 0.9049,
  score: 0.6684,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_003',
  name: 'node_003',
  version: '3.6',
  status: 'degraded',
  priority: 6,
  weight: 0.9857,
  score: 0.9442,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_004',
  name: 'node_004',
  version: '2.6',
  status: 'pending',
  priority: 3,
  weight: 0.6399,
  score: 0.8783,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_005',
  name: 'node_005',
  version: '5.4',
  status: 'recovered',
  priority: 1,
  weight: 0.2292,
  score: 0.7021,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_006',
  name: 'node_006',
  version: '1.0',
  status: 'active',
  priority: 2,
  weight: 0.1994,
  score: 0.7541,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_007',
  name: 'node_007',
  version: '4.7',
  status: 'stable',
  priority: 5,
  weight: 0.8408,
  score: 0.4053,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_008',
  name: 'node_008',
  version: '4.5',
  status: 'recovered',
  priority: 1,
  weight: 0.7803,
  score: 0.8964,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_009',
  name: 'node_009',
  version: '3.8',
  status: 'recovered',
  priority: 6,
  weight: 0.2063,
  score: 0.6581,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_010',
  name: 'node_010',
  version: '4.8',
  status: 'pending',
  priority: 3,
  weight: 0.3052,
  score: 0.2171,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_011',
  name: 'node_011',
  version: '1.5',
  status: 'pending',
  priority: 6,
  weight: 0.6544,
  score: 0.2561,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_012',
  name: 'node_012',
  version: '2.1',
  status: 'degraded',
  priority: 9,
  weight: 0.1406,
  score: 0.7142,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_013',
  name: 'node_013',
  version: '3.2',
  status: 'degraded',
  priority: 4,
  weight: 0.3531,
  score: 0.1738,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_014',
  name: 'node_014',
  version: '4.1',
  status: 'degraded',
  priority: 10,
  weight: 0.8019,
  score: 0.3883,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_015',
  name: 'node_015',
  version: '5.1',
  status: 'degraded',
  priority: 9,
  weight: 0.4776,
  score: 0.4654,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_016',
  name: 'node_016',
  version: '2.0',
  status: 'stable',
  priority: 4,
  weight: 0.8408,
  score: 0.0431,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_017',
  name: 'node_017',
  version: '5.8',
  status: 'stable',
  priority: 5,
  weight: 0.6741,
  score: 0.7804,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_018',
  name: 'node_018',
  version: '5.0',
  status: 'completed',
  priority: 3,
  weight: 0.6209,
  score: 0.7189,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_019',
  name: 'node_019',
  version: '5.3',
  status: 'active',
  priority: 1,
  weight: 0.9497,
  score: 0.6216,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_020',
  name: 'node_020',
  version: '3.9',
  status: 'completed',
  priority: 8,
  weight: 0.6571,
  score: 0.5386,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_021',
  name: 'node_021',
  version: '5.6',
  status: 'completed',
  priority: 8,
  weight: 0.6591,
  score: 0.0873,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_022',
  name: 'node_022',
  version: '3.2',
  status: 'degraded',
  priority: 4,
  weight: 0.711,
  score: 0.0727,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_023',
  name: 'node_023',
  version: '2.6',
  status: 'active',
  priority: 3,
  weight: 0.9992,
  score: 0.8556,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_024',
  name: 'node_024',
  version: '1.8',
  status: 'active',
  priority: 1,
  weight: 0.7768,
  score: 0.8816,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_025',
  name: 'node_025',
  version: '4.0',
  status: 'completed',
  priority: 3,
  weight: 0.4127,
  score: 0.9089,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_026',
  name: 'node_026',
  version: '1.1',
  status: 'failed',
  priority: 2,
  weight: 0.3126,
  score: 0.9722,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_027',
  name: 'node_027',
  version: '3.2',
  status: 'pending',
  priority: 3,
  weight: 0.1509,
  score: 0.4322,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_028',
  name: 'node_028',
  version: '3.1',
  status: 'degraded',
  priority: 1,
  weight: 0.762,
  score: 0.238,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_029',
  name: 'node_029',
  version: '1.6',
  status: 'active',
  priority: 7,
  weight: 0.7739,
  score: 0.6028,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_030',
  name: 'node_030',
  version: '5.8',
  status: 'stable',
  priority: 3,
  weight: 0.7801,
  score: 0.4801,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_031',
  name: 'node_031',
  version: '1.1',
  status: 'stable',
  priority: 5,
  weight: 0.3467,
  score: 0.4728,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_032',
  name: 'node_032',
  version: '3.3',
  status: 'active',
  priority: 5,
  weight: 0.3459,
  score: 0.1396,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_033',
  name: 'node_033',
  version: '5.6',
  status: 'active',
  priority: 3,
  weight: 0.4527,
  score: 0.3494,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_034',
  name: 'node_034',
  version: '1.5',
  status: 'degraded',
  priority: 9,
  weight: 0.4889,
  score: 0.4798,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_035',
  name: 'node_035',
  version: '1.6',
  status: 'pending',
  priority: 5,
  weight: 0.8606,
  score: 0.783,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_036',
  name: 'node_036',
  version: '4.0',
  status: 'stable',
  priority: 7,
  weight: 0.4732,
  score: 0.7722,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_037',
  name: 'node_037',
  version: '5.5',
  status: 'failed',
  priority: 10,
  weight: 0.453,
  score: 0.5562,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_038',
  name: 'node_038',
  version: '5.5',
  status: 'active',
  priority: 10,
  weight: 0.2919,
  score: 0.6288,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_02_state_handlers_1_039',
  name: 'node_039',
  version: '2.4',
  status: 'failed',
  priority: 6,
  weight: 0.2381,
  score: 0.1947,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});
