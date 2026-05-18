:param namespace => 'exceptionrecovery_01_01';
:param batchSize => 128;
:param threshold => 0.368;
:param maxDepth => 4;
:param timeoutSeconds => 109;
:param region => 'eu-west';
:param epoch => 21;
:param version => '1.5.7';

CREATE (n_000:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_000',
  name: 'node_000',
  version: '4.2',
  status: 'pending',
  priority: 8,
  weight: 0.8541,
  score: 0.7139,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_001',
  name: 'node_001',
  version: '5.9',
  status: 'pending',
  priority: 9,
  weight: 0.7657,
  score: 0.7299,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_002',
  name: 'node_002',
  version: '5.8',
  status: 'pending',
  priority: 10,
  weight: 0.6903,
  score: 0.4792,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_003',
  name: 'node_003',
  version: '1.9',
  status: 'active',
  priority: 2,
  weight: 0.6631,
  score: 0.9749,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_004',
  name: 'node_004',
  version: '3.5',
  status: 'failed',
  priority: 4,
  weight: 0.3811,
  score: 0.7672,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_005',
  name: 'node_005',
  version: '1.7',
  status: 'degraded',
  priority: 1,
  weight: 0.5653,
  score: 0.4449,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_006',
  name: 'node_006',
  version: '4.6',
  status: 'failed',
  priority: 10,
  weight: 0.9919,
  score: 0.7154,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_007',
  name: 'node_007',
  version: '5.4',
  status: 'stable',
  priority: 1,
  weight: 0.1012,
  score: 0.3082,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_008',
  name: 'node_008',
  version: '2.4',
  status: 'completed',
  priority: 3,
  weight: 0.2354,
  score: 0.952,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_009',
  name: 'node_009',
  version: '1.0',
  status: 'stable',
  priority: 3,
  weight: 0.2536,
  score: 0.9876,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_010',
  name: 'node_010',
  version: '3.1',
  status: 'completed',
  priority: 1,
  weight: 0.6334,
  score: 0.7548,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_011',
  name: 'node_011',
  version: '1.1',
  status: 'failed',
  priority: 4,
  weight: 0.8892,
  score: 0.1956,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_012',
  name: 'node_012',
  version: '4.8',
  status: 'active',
  priority: 7,
  weight: 0.1295,
  score: 0.4812,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_013',
  name: 'node_013',
  version: '3.4',
  status: 'recovered',
  priority: 1,
  weight: 0.9593,
  score: 0.1255,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_014',
  name: 'node_014',
  version: '2.9',
  status: 'degraded',
  priority: 8,
  weight: 0.4395,
  score: 0.8912,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_015',
  name: 'node_015',
  version: '1.2',
  status: 'failed',
  priority: 2,
  weight: 0.7327,
  score: 0.4993,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_016',
  name: 'node_016',
  version: '3.5',
  status: 'stable',
  priority: 10,
  weight: 0.4021,
  score: 0.2596,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_017',
  name: 'node_017',
  version: '5.7',
  status: 'stable',
  priority: 9,
  weight: 0.8907,
  score: 0.109,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_018',
  name: 'node_018',
  version: '2.0',
  status: 'failed',
  priority: 4,
  weight: 0.6806,
  score: 0.7579,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 24,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_019',
  name: 'node_019',
  version: '3.8',
  status: 'degraded',
  priority: 10,
  weight: 0.467,
  score: 0.3322,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_020',
  name: 'node_020',
  version: '1.3',
  status: 'recovered',
  priority: 5,
  weight: 0.8265,
  score: 0.9057,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_021',
  name: 'node_021',
  version: '4.4',
  status: 'stable',
  priority: 10,
  weight: 0.6823,
  score: 0.3854,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_022',
  name: 'node_022',
  version: '1.4',
  status: 'stable',
  priority: 9,
  weight: 0.7357,
  score: 0.9095,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_023',
  name: 'node_023',
  version: '1.1',
  status: 'completed',
  priority: 10,
  weight: 0.217,
  score: 0.6126,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_024',
  name: 'node_024',
  version: '2.9',
  status: 'degraded',
  priority: 9,
  weight: 0.9308,
  score: 0.7808,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_025',
  name: 'node_025',
  version: '5.6',
  status: 'completed',
  priority: 2,
  weight: 0.2404,
  score: 0.3616,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_026',
  name: 'node_026',
  version: '1.2',
  status: 'degraded',
  priority: 3,
  weight: 0.9831,
  score: 0.2211,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_027',
  name: 'node_027',
  version: '4.4',
  status: 'recovered',
  priority: 6,
  weight: 0.6324,
  score: 0.6863,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_028',
  name: 'node_028',
  version: '1.9',
  status: 'stable',
  priority: 8,
  weight: 0.575,
  score: 0.4301,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_029',
  name: 'node_029',
  version: '2.6',
  status: 'failed',
  priority: 4,
  weight: 0.1694,
  score: 0.7829,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_030',
  name: 'node_030',
  version: '5.8',
  status: 'active',
  priority: 1,
  weight: 0.7112,
  score: 0.5256,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_031',
  name: 'node_031',
  version: '4.7',
  status: 'stable',
  priority: 4,
  weight: 0.5361,
  score: 0.1878,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_032',
  name: 'node_032',
  version: '5.9',
  status: 'failed',
  priority: 9,
  weight: 0.8013,
  score: 0.1696,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_033',
  name: 'node_033',
  version: '4.1',
  status: 'recovered',
  priority: 10,
  weight: 0.8789,
  score: 0.2414,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_034',
  name: 'node_034',
  version: '4.7',
  status: 'active',
  priority: 9,
  weight: 0.1462,
  score: 0.8194,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_035',
  name: 'node_035',
  version: '1.7',
  status: 'degraded',
  priority: 7,
  weight: 0.234,
  score: 0.8917,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_036',
  name: 'node_036',
  version: '1.9',
  status: 'active',
  priority: 8,
  weight: 0.4079,
  score: 0.2055,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_037',
  name: 'node_037',
  version: '5.5',
  status: 'active',
  priority: 8,
  weight: 0.4408,
  score: 0.8723,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_038',
  name: 'node_038',
  version: '4.6',
  status: 'recovered',
  priority: 5,
  weight: 0.522,
  score: 0.4101,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:ExceptionRecovery:Node {
  identifier: 'exceptionrecovery_03_config_managers_1_039',
  name: 'node_039',
  version: '4.2',
  status: 'completed',
  priority: 7,
  weight: 0.8987,
  score: 0.2288,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: false
});
