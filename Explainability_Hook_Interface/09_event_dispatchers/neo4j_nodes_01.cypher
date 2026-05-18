:param namespace => 'explainability_01_01';
:param batchSize => 512;
:param threshold => 0.585;
:param maxDepth => 12;
:param timeoutSeconds => 31;
:param region => 'ap-south';
:param epoch => 12;
:param version => '5.6.5';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_000',
  name: 'node_000',
  version: '2.3',
  status: 'pending',
  priority: 5,
  weight: 0.3244,
  score: 0.2377,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_001',
  name: 'node_001',
  version: '5.5',
  status: 'failed',
  priority: 1,
  weight: 0.5244,
  score: 0.9647,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_002',
  name: 'node_002',
  version: '4.8',
  status: 'failed',
  priority: 8,
  weight: 0.6437,
  score: 0.5259,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_003',
  name: 'node_003',
  version: '2.7',
  status: 'degraded',
  priority: 2,
  weight: 0.4547,
  score: 0.9651,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_004',
  name: 'node_004',
  version: '3.5',
  status: 'failed',
  priority: 7,
  weight: 0.2206,
  score: 0.6036,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_005',
  name: 'node_005',
  version: '4.2',
  status: 'failed',
  priority: 6,
  weight: 0.3673,
  score: 0.2512,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_006',
  name: 'node_006',
  version: '5.9',
  status: 'completed',
  priority: 8,
  weight: 0.2561,
  score: 0.7733,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_007',
  name: 'node_007',
  version: '4.9',
  status: 'failed',
  priority: 3,
  weight: 0.7126,
  score: 0.3593,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_008',
  name: 'node_008',
  version: '2.5',
  status: 'failed',
  priority: 3,
  weight: 0.6148,
  score: 0.3368,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_009',
  name: 'node_009',
  version: '2.0',
  status: 'completed',
  priority: 8,
  weight: 0.1658,
  score: 0.125,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_010',
  name: 'node_010',
  version: '5.7',
  status: 'degraded',
  priority: 2,
  weight: 0.9274,
  score: 0.6211,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_011',
  name: 'node_011',
  version: '5.1',
  status: 'recovered',
  priority: 8,
  weight: 0.1187,
  score: 0.3748,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_012',
  name: 'node_012',
  version: '1.0',
  status: 'recovered',
  priority: 7,
  weight: 0.4296,
  score: 0.5489,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_013',
  name: 'node_013',
  version: '5.9',
  status: 'completed',
  priority: 6,
  weight: 0.6288,
  score: 0.9198,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_014',
  name: 'node_014',
  version: '3.9',
  status: 'degraded',
  priority: 8,
  weight: 0.2228,
  score: 0.2729,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_015',
  name: 'node_015',
  version: '3.2',
  status: 'degraded',
  priority: 2,
  weight: 0.1464,
  score: 0.4447,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_016',
  name: 'node_016',
  version: '5.8',
  status: 'active',
  priority: 7,
  weight: 0.4819,
  score: 0.9245,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_017',
  name: 'node_017',
  version: '2.1',
  status: 'recovered',
  priority: 3,
  weight: 0.2457,
  score: 0.0335,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_018',
  name: 'node_018',
  version: '1.7',
  status: 'active',
  priority: 10,
  weight: 0.2551,
  score: 0.319,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_019',
  name: 'node_019',
  version: '5.2',
  status: 'pending',
  priority: 10,
  weight: 0.2789,
  score: 0.7882,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_020',
  name: 'node_020',
  version: '3.3',
  status: 'pending',
  priority: 1,
  weight: 0.8231,
  score: 0.9498,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_021',
  name: 'node_021',
  version: '5.1',
  status: 'pending',
  priority: 9,
  weight: 0.8553,
  score: 0.1065,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_022',
  name: 'node_022',
  version: '3.2',
  status: 'completed',
  priority: 3,
  weight: 0.8645,
  score: 0.765,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_023',
  name: 'node_023',
  version: '1.9',
  status: 'degraded',
  priority: 9,
  weight: 0.7201,
  score: 0.2719,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_024',
  name: 'node_024',
  version: '1.1',
  status: 'failed',
  priority: 5,
  weight: 0.8372,
  score: 0.0662,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_025',
  name: 'node_025',
  version: '3.3',
  status: 'degraded',
  priority: 3,
  weight: 0.2831,
  score: 0.8434,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_026',
  name: 'node_026',
  version: '2.4',
  status: 'recovered',
  priority: 6,
  weight: 0.4105,
  score: 0.3457,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_027',
  name: 'node_027',
  version: '5.0',
  status: 'completed',
  priority: 5,
  weight: 0.6957,
  score: 0.1523,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_028',
  name: 'node_028',
  version: '5.6',
  status: 'stable',
  priority: 7,
  weight: 0.3843,
  score: 0.6963,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_029',
  name: 'node_029',
  version: '4.9',
  status: 'active',
  priority: 2,
  weight: 0.5036,
  score: 0.8175,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_030',
  name: 'node_030',
  version: '2.2',
  status: 'failed',
  priority: 4,
  weight: 0.3451,
  score: 0.8239,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_031',
  name: 'node_031',
  version: '2.4',
  status: 'recovered',
  priority: 6,
  weight: 0.5701,
  score: 0.0411,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_032',
  name: 'node_032',
  version: '1.3',
  status: 'active',
  priority: 1,
  weight: 0.1529,
  score: 0.3573,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_033',
  name: 'node_033',
  version: '1.4',
  status: 'completed',
  priority: 10,
  weight: 0.1897,
  score: 0.8281,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_034',
  name: 'node_034',
  version: '2.9',
  status: 'recovered',
  priority: 2,
  weight: 0.4416,
  score: 0.6214,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_035',
  name: 'node_035',
  version: '3.4',
  status: 'recovered',
  priority: 2,
  weight: 0.4826,
  score: 0.4636,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_036',
  name: 'node_036',
  version: '4.7',
  status: 'completed',
  priority: 9,
  weight: 0.1875,
  score: 0.8957,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_037',
  name: 'node_037',
  version: '1.1',
  status: 'degraded',
  priority: 7,
  weight: 0.6732,
  score: 0.4609,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 79,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_038',
  name: 'node_038',
  version: '1.5',
  status: 'active',
  priority: 6,
  weight: 0.5904,
  score: 0.6725,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_09_event_dispatchers_1_039',
  name: 'node_039',
  version: '3.6',
  status: 'failed',
  priority: 1,
  weight: 0.2166,
  score: 0.803,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: false
});
