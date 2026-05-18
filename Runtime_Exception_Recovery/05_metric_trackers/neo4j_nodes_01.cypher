:param namespace => 'exceptionrecovery_01_01';
:param batchSize => 32;
:param threshold => 0.62;
:param maxDepth => 11;
:param timeoutSeconds => 55;
:param region => 'us-east';
:param epoch => 23;
:param version => '3.1.8';

CREATE (n_000:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_000',
  name: 'node_000',
  version: '1.9',
  status: 'recovered',
  priority: 5,
  weight: 0.9604,
  score: 0.0052,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_001',
  name: 'node_001',
  version: '2.7',
  status: 'recovered',
  priority: 3,
  weight: 0.2686,
  score: 0.4874,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_002',
  name: 'node_002',
  version: '1.4',
  status: 'failed',
  priority: 6,
  weight: 0.688,
  score: 0.3694,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_003',
  name: 'node_003',
  version: '3.8',
  status: 'completed',
  priority: 10,
  weight: 0.9723,
  score: 0.8042,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_004',
  name: 'node_004',
  version: '2.3',
  status: 'pending',
  priority: 5,
  weight: 0.5554,
  score: 0.4067,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_005',
  name: 'node_005',
  version: '5.5',
  status: 'pending',
  priority: 7,
  weight: 0.9542,
  score: 0.5824,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_006',
  name: 'node_006',
  version: '1.6',
  status: 'completed',
  priority: 9,
  weight: 0.2291,
  score: 0.3629,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_007',
  name: 'node_007',
  version: '4.7',
  status: 'active',
  priority: 1,
  weight: 0.1774,
  score: 0.8501,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_008',
  name: 'node_008',
  version: '3.6',
  status: 'degraded',
  priority: 8,
  weight: 0.1851,
  score: 0.3481,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_009',
  name: 'node_009',
  version: '4.2',
  status: 'pending',
  priority: 1,
  weight: 0.2089,
  score: 0.5693,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_010',
  name: 'node_010',
  version: '5.1',
  status: 'pending',
  priority: 5,
  weight: 0.723,
  score: 0.6757,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_011',
  name: 'node_011',
  version: '4.8',
  status: 'failed',
  priority: 5,
  weight: 0.3742,
  score: 0.1262,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_012',
  name: 'node_012',
  version: '1.5',
  status: 'completed',
  priority: 6,
  weight: 0.3771,
  score: 0.0845,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_013',
  name: 'node_013',
  version: '2.9',
  status: 'completed',
  priority: 3,
  weight: 0.6142,
  score: 0.4192,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_014',
  name: 'node_014',
  version: '2.0',
  status: 'recovered',
  priority: 3,
  weight: 0.2978,
  score: 0.182,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_015',
  name: 'node_015',
  version: '5.8',
  status: 'pending',
  priority: 1,
  weight: 0.9741,
  score: 0.6638,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_016',
  name: 'node_016',
  version: '2.0',
  status: 'failed',
  priority: 1,
  weight: 0.1227,
  score: 0.7155,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_017',
  name: 'node_017',
  version: '3.7',
  status: 'degraded',
  priority: 6,
  weight: 0.6089,
  score: 0.0651,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_018',
  name: 'node_018',
  version: '2.1',
  status: 'pending',
  priority: 3,
  weight: 0.8995,
  score: 0.346,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_019',
  name: 'node_019',
  version: '3.4',
  status: 'active',
  priority: 2,
  weight: 0.5947,
  score: 0.3427,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_020',
  name: 'node_020',
  version: '3.4',
  status: 'active',
  priority: 7,
  weight: 0.3726,
  score: 0.4845,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_021',
  name: 'node_021',
  version: '3.1',
  status: 'recovered',
  priority: 5,
  weight: 0.5943,
  score: 0.3909,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_022',
  name: 'node_022',
  version: '2.8',
  status: 'failed',
  priority: 5,
  weight: 0.7781,
  score: 0.8675,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_023',
  name: 'node_023',
  version: '1.5',
  status: 'failed',
  priority: 10,
  weight: 0.8524,
  score: 0.1624,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_024',
  name: 'node_024',
  version: '5.3',
  status: 'recovered',
  priority: 7,
  weight: 0.5588,
  score: 0.2037,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_025',
  name: 'node_025',
  version: '3.0',
  status: 'pending',
  priority: 10,
  weight: 0.5089,
  score: 0.892,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_026',
  name: 'node_026',
  version: '3.2',
  status: 'active',
  priority: 8,
  weight: 0.7141,
  score: 0.0132,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_027',
  name: 'node_027',
  version: '4.7',
  status: 'active',
  priority: 5,
  weight: 0.6022,
  score: 0.0699,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_028',
  name: 'node_028',
  version: '4.2',
  status: 'completed',
  priority: 5,
  weight: 0.9528,
  score: 0.0826,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_029',
  name: 'node_029',
  version: '2.0',
  status: 'stable',
  priority: 9,
  weight: 0.3459,
  score: 0.23,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_030',
  name: 'node_030',
  version: '4.9',
  status: 'pending',
  priority: 6,
  weight: 0.505,
  score: 0.5493,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_031',
  name: 'node_031',
  version: '2.3',
  status: 'failed',
  priority: 6,
  weight: 0.86,
  score: 0.2981,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_032',
  name: 'node_032',
  version: '3.5',
  status: 'failed',
  priority: 7,
  weight: 0.3988,
  score: 0.1737,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_033',
  name: 'node_033',
  version: '5.6',
  status: 'degraded',
  priority: 7,
  weight: 0.5147,
  score: 0.329,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_034',
  name: 'node_034',
  version: '4.8',
  status: 'stable',
  priority: 7,
  weight: 0.8731,
  score: 0.4018,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_035',
  name: 'node_035',
  version: '5.1',
  status: 'completed',
  priority: 1,
  weight: 0.9263,
  score: 0.6303,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_036',
  name: 'node_036',
  version: '2.7',
  status: 'pending',
  priority: 9,
  weight: 0.4305,
  score: 0.9526,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_037',
  name: 'node_037',
  version: '3.7',
  status: 'recovered',
  priority: 8,
  weight: 0.767,
  score: 0.4084,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_038',
  name: 'node_038',
  version: '5.1',
  status: 'stable',
  priority: 10,
  weight: 0.8738,
  score: 0.6074,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_05_metric_trackers_1_039',
  name: 'node_039',
  version: '3.0',
  status: 'degraded',
  priority: 4,
  weight: 0.1768,
  score: 0.1257,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});
