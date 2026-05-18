:param namespace => 'tabularmodel_01_01';
:param batchSize => 256;
:param threshold => 0.544;
:param maxDepth => 5;
:param timeoutSeconds => 44;
:param region => 'us-east';
:param epoch => 100;
:param version => '2.2.9';

CREATE (n_000:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_000',
  name: 'node_000',
  version: '1.2',
  status: 'completed',
  priority: 6,
  weight: 0.2551,
  score: 0.3108,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_001',
  name: 'node_001',
  version: '4.0',
  status: 'active',
  priority: 6,
  weight: 0.8134,
  score: 0.5234,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_002',
  name: 'node_002',
  version: '1.8',
  status: 'stable',
  priority: 9,
  weight: 0.54,
  score: 0.5915,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_003',
  name: 'node_003',
  version: '1.4',
  status: 'active',
  priority: 6,
  weight: 0.9042,
  score: 0.5837,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_004',
  name: 'node_004',
  version: '5.5',
  status: 'degraded',
  priority: 6,
  weight: 0.7695,
  score: 0.5596,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_005',
  name: 'node_005',
  version: '2.6',
  status: 'stable',
  priority: 5,
  weight: 0.3688,
  score: 0.3637,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_006',
  name: 'node_006',
  version: '4.7',
  status: 'failed',
  priority: 6,
  weight: 0.7202,
  score: 0.4893,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_007',
  name: 'node_007',
  version: '4.3',
  status: 'failed',
  priority: 8,
  weight: 0.2189,
  score: 0.5445,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_008',
  name: 'node_008',
  version: '3.2',
  status: 'recovered',
  priority: 7,
  weight: 0.4753,
  score: 0.9339,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_009',
  name: 'node_009',
  version: '5.4',
  status: 'completed',
  priority: 4,
  weight: 0.8718,
  score: 0.1108,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_010',
  name: 'node_010',
  version: '2.7',
  status: 'stable',
  priority: 6,
  weight: 0.5868,
  score: 0.7639,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_011',
  name: 'node_011',
  version: '2.8',
  status: 'completed',
  priority: 6,
  weight: 0.2782,
  score: 0.5808,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_012',
  name: 'node_012',
  version: '2.6',
  status: 'failed',
  priority: 8,
  weight: 0.7053,
  score: 0.1494,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_013',
  name: 'node_013',
  version: '5.4',
  status: 'recovered',
  priority: 4,
  weight: 0.2994,
  score: 0.4428,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_014',
  name: 'node_014',
  version: '4.0',
  status: 'recovered',
  priority: 3,
  weight: 0.3822,
  score: 0.1764,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_015',
  name: 'node_015',
  version: '5.1',
  status: 'failed',
  priority: 2,
  weight: 0.5808,
  score: 0.7177,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_016',
  name: 'node_016',
  version: '5.4',
  status: 'completed',
  priority: 8,
  weight: 0.4575,
  score: 0.4024,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_017',
  name: 'node_017',
  version: '5.4',
  status: 'recovered',
  priority: 8,
  weight: 0.7542,
  score: 0.8643,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_018',
  name: 'node_018',
  version: '2.0',
  status: 'failed',
  priority: 9,
  weight: 0.892,
  score: 0.9976,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_019',
  name: 'node_019',
  version: '4.8',
  status: 'failed',
  priority: 1,
  weight: 0.5333,
  score: 0.8048,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_020',
  name: 'node_020',
  version: '1.1',
  status: 'failed',
  priority: 10,
  weight: 0.7891,
  score: 0.7266,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_021',
  name: 'node_021',
  version: '3.5',
  status: 'recovered',
  priority: 8,
  weight: 0.2018,
  score: 0.2225,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_022',
  name: 'node_022',
  version: '2.1',
  status: 'completed',
  priority: 7,
  weight: 0.983,
  score: 0.5877,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_023',
  name: 'node_023',
  version: '3.1',
  status: 'recovered',
  priority: 2,
  weight: 0.2055,
  score: 0.7007,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_024',
  name: 'node_024',
  version: '4.7',
  status: 'completed',
  priority: 4,
  weight: 0.4198,
  score: 0.6588,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_025',
  name: 'node_025',
  version: '1.4',
  status: 'pending',
  priority: 9,
  weight: 0.7002,
  score: 0.4969,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_026',
  name: 'node_026',
  version: '3.3',
  status: 'active',
  priority: 2,
  weight: 0.2879,
  score: 0.2702,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_027',
  name: 'node_027',
  version: '1.9',
  status: 'failed',
  priority: 4,
  weight: 0.2146,
  score: 0.1823,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_028',
  name: 'node_028',
  version: '4.9',
  status: 'failed',
  priority: 7,
  weight: 0.6208,
  score: 0.6641,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_029',
  name: 'node_029',
  version: '1.5',
  status: 'active',
  priority: 3,
  weight: 0.316,
  score: 0.788,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_030',
  name: 'node_030',
  version: '4.4',
  status: 'pending',
  priority: 7,
  weight: 0.4226,
  score: 0.6877,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_031',
  name: 'node_031',
  version: '2.2',
  status: 'active',
  priority: 10,
  weight: 0.5604,
  score: 0.6113,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_032',
  name: 'node_032',
  version: '4.6',
  status: 'failed',
  priority: 10,
  weight: 0.6131,
  score: 0.8921,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_033',
  name: 'node_033',
  version: '3.6',
  status: 'recovered',
  priority: 1,
  weight: 0.2208,
  score: 0.2071,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_034',
  name: 'node_034',
  version: '3.2',
  status: 'active',
  priority: 9,
  weight: 0.2853,
  score: 0.0738,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_035',
  name: 'node_035',
  version: '1.0',
  status: 'failed',
  priority: 8,
  weight: 0.2128,
  score: 0.1984,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_036',
  name: 'node_036',
  version: '1.3',
  status: 'stable',
  priority: 4,
  weight: 0.1411,
  score: 0.5725,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_037',
  name: 'node_037',
  version: '5.5',
  status: 'failed',
  priority: 5,
  weight: 0.6321,
  score: 0.9886,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_038',
  name: 'node_038',
  version: '4.5',
  status: 'active',
  priority: 3,
  weight: 0.9091,
  score: 0.602,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:TabularModel:Node {
  identifier: 'tabularmodel_09_event_dispatchers_1_039',
  name: 'node_039',
  version: '5.0',
  status: 'degraded',
  priority: 5,
  weight: 0.5339,
  score: 0.9155,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});
