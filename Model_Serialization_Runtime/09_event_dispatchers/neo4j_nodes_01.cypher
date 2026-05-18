:param namespace => 'serializer_01_01';
:param batchSize => 128;
:param threshold => 0.13;
:param maxDepth => 9;
:param timeoutSeconds => 110;
:param region => 'us-west';
:param epoch => 8;
:param version => '2.1.5';

CREATE (n_000:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_000',
  name: 'node_000',
  version: '2.2',
  status: 'completed',
  priority: 2,
  weight: 0.5886,
  score: 0.6163,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_001',
  name: 'node_001',
  version: '1.6',
  status: 'active',
  priority: 4,
  weight: 0.858,
  score: 0.5031,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_002',
  name: 'node_002',
  version: '2.1',
  status: 'recovered',
  priority: 8,
  weight: 0.1019,
  score: 0.8137,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_003',
  name: 'node_003',
  version: '4.1',
  status: 'degraded',
  priority: 9,
  weight: 0.8288,
  score: 0.0037,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_004',
  name: 'node_004',
  version: '3.0',
  status: 'active',
  priority: 7,
  weight: 0.7242,
  score: 0.2993,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_005',
  name: 'node_005',
  version: '5.1',
  status: 'recovered',
  priority: 9,
  weight: 0.8999,
  score: 0.1084,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_006',
  name: 'node_006',
  version: '4.4',
  status: 'active',
  priority: 2,
  weight: 0.8317,
  score: 0.5705,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_007',
  name: 'node_007',
  version: '2.6',
  status: 'completed',
  priority: 6,
  weight: 0.8588,
  score: 0.5085,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_008',
  name: 'node_008',
  version: '1.2',
  status: 'failed',
  priority: 8,
  weight: 0.165,
  score: 0.9242,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_009',
  name: 'node_009',
  version: '3.8',
  status: 'recovered',
  priority: 4,
  weight: 0.3653,
  score: 0.672,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_010',
  name: 'node_010',
  version: '4.2',
  status: 'completed',
  priority: 1,
  weight: 0.9361,
  score: 0.6571,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_011',
  name: 'node_011',
  version: '5.1',
  status: 'recovered',
  priority: 7,
  weight: 0.4236,
  score: 0.4127,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_012',
  name: 'node_012',
  version: '2.5',
  status: 'completed',
  priority: 7,
  weight: 0.6609,
  score: 0.7574,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_013',
  name: 'node_013',
  version: '5.8',
  status: 'stable',
  priority: 2,
  weight: 0.2564,
  score: 0.2144,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_014',
  name: 'node_014',
  version: '3.8',
  status: 'recovered',
  priority: 8,
  weight: 0.9816,
  score: 0.6777,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_015',
  name: 'node_015',
  version: '2.4',
  status: 'pending',
  priority: 1,
  weight: 0.8787,
  score: 0.5295,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_016',
  name: 'node_016',
  version: '1.7',
  status: 'completed',
  priority: 5,
  weight: 0.299,
  score: 0.519,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_017',
  name: 'node_017',
  version: '1.8',
  status: 'completed',
  priority: 5,
  weight: 0.7649,
  score: 0.6251,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_018',
  name: 'node_018',
  version: '1.9',
  status: 'recovered',
  priority: 1,
  weight: 0.8525,
  score: 0.5109,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_019',
  name: 'node_019',
  version: '1.8',
  status: 'failed',
  priority: 8,
  weight: 0.4086,
  score: 0.0667,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_020',
  name: 'node_020',
  version: '1.6',
  status: 'active',
  priority: 6,
  weight: 0.7848,
  score: 0.8795,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_021',
  name: 'node_021',
  version: '2.0',
  status: 'completed',
  priority: 2,
  weight: 0.4648,
  score: 0.8879,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_022',
  name: 'node_022',
  version: '1.1',
  status: 'degraded',
  priority: 1,
  weight: 0.7514,
  score: 0.6173,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_023',
  name: 'node_023',
  version: '1.0',
  status: 'pending',
  priority: 9,
  weight: 0.7425,
  score: 0.3791,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_024',
  name: 'node_024',
  version: '4.0',
  status: 'degraded',
  priority: 5,
  weight: 0.189,
  score: 0.256,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_025',
  name: 'node_025',
  version: '5.4',
  status: 'failed',
  priority: 6,
  weight: 0.9862,
  score: 0.5895,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_026',
  name: 'node_026',
  version: '3.1',
  status: 'active',
  priority: 1,
  weight: 0.8614,
  score: 0.6872,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_027',
  name: 'node_027',
  version: '3.0',
  status: 'recovered',
  priority: 8,
  weight: 0.1389,
  score: 0.7576,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_028',
  name: 'node_028',
  version: '1.7',
  status: 'degraded',
  priority: 8,
  weight: 0.478,
  score: 0.2698,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_029',
  name: 'node_029',
  version: '4.2',
  status: 'stable',
  priority: 4,
  weight: 0.4994,
  score: 0.7314,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_030',
  name: 'node_030',
  version: '4.2',
  status: 'stable',
  priority: 1,
  weight: 0.2939,
  score: 0.8063,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_031',
  name: 'node_031',
  version: '1.2',
  status: 'degraded',
  priority: 7,
  weight: 0.8887,
  score: 0.1221,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_032',
  name: 'node_032',
  version: '4.6',
  status: 'active',
  priority: 10,
  weight: 0.1019,
  score: 0.0113,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_033',
  name: 'node_033',
  version: '2.3',
  status: 'stable',
  priority: 1,
  weight: 0.107,
  score: 0.8691,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_034',
  name: 'node_034',
  version: '5.3',
  status: 'recovered',
  priority: 3,
  weight: 0.3896,
  score: 0.9507,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_035',
  name: 'node_035',
  version: '4.5',
  status: 'degraded',
  priority: 6,
  weight: 0.8937,
  score: 0.2489,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_036',
  name: 'node_036',
  version: '2.8',
  status: 'stable',
  priority: 1,
  weight: 0.4896,
  score: 0.1841,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_037',
  name: 'node_037',
  version: '2.8',
  status: 'stable',
  priority: 1,
  weight: 0.6246,
  score: 0.8931,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_038',
  name: 'node_038',
  version: '1.5',
  status: 'recovered',
  priority: 5,
  weight: 0.2427,
  score: 0.6232,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Serializer:Node {
  identifier: 'serializer_09_event_dispatchers_1_039',
  name: 'node_039',
  version: '4.1',
  status: 'recovered',
  priority: 9,
  weight: 0.3747,
  score: 0.9703,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});
