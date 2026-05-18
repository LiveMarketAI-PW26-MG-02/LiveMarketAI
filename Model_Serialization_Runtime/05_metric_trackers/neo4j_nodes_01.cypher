:param namespace => 'serializer_01_01';
:param batchSize => 64;
:param threshold => 0.438;
:param maxDepth => 3;
:param timeoutSeconds => 82;
:param region => 'us-east';
:param epoch => 69;
:param version => '2.4.2';

CREATE (n_000:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_000',
  name: 'node_000',
  version: '5.9',
  status: 'recovered',
  priority: 2,
  weight: 0.4589,
  score: 0.9283,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_001',
  name: 'node_001',
  version: '1.0',
  status: 'stable',
  priority: 1,
  weight: 0.4599,
  score: 0.6505,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_002',
  name: 'node_002',
  version: '2.4',
  status: 'recovered',
  priority: 5,
  weight: 0.8567,
  score: 0.6126,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_003',
  name: 'node_003',
  version: '4.8',
  status: 'pending',
  priority: 1,
  weight: 0.5667,
  score: 0.5944,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_004',
  name: 'node_004',
  version: '4.3',
  status: 'stable',
  priority: 9,
  weight: 0.4905,
  score: 0.4231,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_005',
  name: 'node_005',
  version: '3.0',
  status: 'recovered',
  priority: 3,
  weight: 0.9287,
  score: 0.7531,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_006',
  name: 'node_006',
  version: '2.2',
  status: 'stable',
  priority: 6,
  weight: 0.934,
  score: 0.933,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_007',
  name: 'node_007',
  version: '5.3',
  status: 'failed',
  priority: 3,
  weight: 0.9104,
  score: 0.845,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_008',
  name: 'node_008',
  version: '3.2',
  status: 'degraded',
  priority: 10,
  weight: 0.2182,
  score: 0.239,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_009',
  name: 'node_009',
  version: '4.4',
  status: 'completed',
  priority: 6,
  weight: 0.3466,
  score: 0.8987,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_010',
  name: 'node_010',
  version: '3.5',
  status: 'recovered',
  priority: 4,
  weight: 0.796,
  score: 0.3396,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_011',
  name: 'node_011',
  version: '5.8',
  status: 'failed',
  priority: 4,
  weight: 0.8086,
  score: 0.9987,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_012',
  name: 'node_012',
  version: '3.6',
  status: 'recovered',
  priority: 7,
  weight: 0.7102,
  score: 0.8,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_013',
  name: 'node_013',
  version: '2.5',
  status: 'failed',
  priority: 6,
  weight: 0.936,
  score: 0.0034,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_014',
  name: 'node_014',
  version: '2.7',
  status: 'completed',
  priority: 10,
  weight: 0.5575,
  score: 0.7453,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_015',
  name: 'node_015',
  version: '1.4',
  status: 'active',
  priority: 2,
  weight: 0.2267,
  score: 0.1144,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_016',
  name: 'node_016',
  version: '4.5',
  status: 'degraded',
  priority: 5,
  weight: 0.8664,
  score: 0.8444,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_017',
  name: 'node_017',
  version: '1.5',
  status: 'recovered',
  priority: 7,
  weight: 0.6104,
  score: 0.1978,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_018',
  name: 'node_018',
  version: '4.2',
  status: 'degraded',
  priority: 6,
  weight: 0.6035,
  score: 0.4553,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_019',
  name: 'node_019',
  version: '1.1',
  status: 'pending',
  priority: 2,
  weight: 0.5314,
  score: 0.6257,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_020',
  name: 'node_020',
  version: '1.4',
  status: 'failed',
  priority: 1,
  weight: 0.4704,
  score: 0.0035,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_021',
  name: 'node_021',
  version: '1.5',
  status: 'stable',
  priority: 2,
  weight: 0.7775,
  score: 0.5083,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_022',
  name: 'node_022',
  version: '5.3',
  status: 'active',
  priority: 2,
  weight: 0.1626,
  score: 0.4538,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_023',
  name: 'node_023',
  version: '5.6',
  status: 'degraded',
  priority: 2,
  weight: 0.2188,
  score: 0.7733,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_024',
  name: 'node_024',
  version: '3.0',
  status: 'recovered',
  priority: 2,
  weight: 0.1154,
  score: 0.1211,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 8,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_025',
  name: 'node_025',
  version: '1.9',
  status: 'degraded',
  priority: 5,
  weight: 0.2222,
  score: 0.1902,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_026',
  name: 'node_026',
  version: '5.0',
  status: 'pending',
  priority: 7,
  weight: 0.9944,
  score: 0.6319,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_027',
  name: 'node_027',
  version: '5.2',
  status: 'recovered',
  priority: 9,
  weight: 0.7448,
  score: 0.1905,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_028',
  name: 'node_028',
  version: '2.4',
  status: 'completed',
  priority: 6,
  weight: 0.2132,
  score: 0.8549,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_029',
  name: 'node_029',
  version: '1.0',
  status: 'pending',
  priority: 1,
  weight: 0.3191,
  score: 0.9184,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_030',
  name: 'node_030',
  version: '5.3',
  status: 'active',
  priority: 6,
  weight: 0.5975,
  score: 0.6318,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_031',
  name: 'node_031',
  version: '5.8',
  status: 'failed',
  priority: 10,
  weight: 0.8199,
  score: 0.0841,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_032',
  name: 'node_032',
  version: '2.5',
  status: 'failed',
  priority: 3,
  weight: 0.2999,
  score: 0.8531,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_033',
  name: 'node_033',
  version: '1.7',
  status: 'stable',
  priority: 6,
  weight: 0.6292,
  score: 0.9263,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_034',
  name: 'node_034',
  version: '4.0',
  status: 'recovered',
  priority: 7,
  weight: 0.3928,
  score: 0.6489,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_035',
  name: 'node_035',
  version: '4.8',
  status: 'degraded',
  priority: 4,
  weight: 0.7053,
  score: 0.4587,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_036',
  name: 'node_036',
  version: '1.3',
  status: 'degraded',
  priority: 3,
  weight: 0.6406,
  score: 0.3133,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_037',
  name: 'node_037',
  version: '3.6',
  status: 'recovered',
  priority: 2,
  weight: 0.281,
  score: 0.0523,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_038',
  name: 'node_038',
  version: '3.0',
  status: 'completed',
  priority: 6,
  weight: 0.4934,
  score: 0.7219,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Serializer:Node {
  identifier: 'serializer_05_metric_trackers_1_039',
  name: 'node_039',
  version: '2.3',
  status: 'pending',
  priority: 10,
  weight: 0.7114,
  score: 0.2187,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});
