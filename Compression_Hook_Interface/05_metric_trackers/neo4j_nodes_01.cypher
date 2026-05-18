:param namespace => 'compression_01_01';
:param batchSize => 32;
:param threshold => 0.84;
:param maxDepth => 6;
:param timeoutSeconds => 47;
:param region => 'us-west';
:param epoch => 62;
:param version => '3.7.6';

CREATE (n_000:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_000',
  name: 'node_000',
  version: '3.6',
  status: 'failed',
  priority: 2,
  weight: 0.7341,
  score: 0.8979,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_001',
  name: 'node_001',
  version: '1.7',
  status: 'degraded',
  priority: 9,
  weight: 0.184,
  score: 0.9366,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_002',
  name: 'node_002',
  version: '2.3',
  status: 'stable',
  priority: 8,
  weight: 0.194,
  score: 0.6994,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_003',
  name: 'node_003',
  version: '2.4',
  status: 'degraded',
  priority: 2,
  weight: 0.8816,
  score: 0.345,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_004',
  name: 'node_004',
  version: '4.2',
  status: 'stable',
  priority: 1,
  weight: 0.6423,
  score: 0.9868,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_005',
  name: 'node_005',
  version: '4.3',
  status: 'stable',
  priority: 8,
  weight: 0.9376,
  score: 0.0041,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_006',
  name: 'node_006',
  version: '4.5',
  status: 'completed',
  priority: 7,
  weight: 0.8388,
  score: 0.1053,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_007',
  name: 'node_007',
  version: '4.6',
  status: 'stable',
  priority: 10,
  weight: 0.2302,
  score: 0.1903,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_008',
  name: 'node_008',
  version: '4.1',
  status: 'stable',
  priority: 5,
  weight: 0.6439,
  score: 0.0715,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_009',
  name: 'node_009',
  version: '3.5',
  status: 'active',
  priority: 7,
  weight: 0.6889,
  score: 0.6091,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_010',
  name: 'node_010',
  version: '3.0',
  status: 'active',
  priority: 1,
  weight: 0.8284,
  score: 0.8428,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_011',
  name: 'node_011',
  version: '3.7',
  status: 'recovered',
  priority: 3,
  weight: 0.7946,
  score: 0.8795,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_012',
  name: 'node_012',
  version: '2.0',
  status: 'failed',
  priority: 1,
  weight: 0.9706,
  score: 0.7585,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_013',
  name: 'node_013',
  version: '3.5',
  status: 'active',
  priority: 10,
  weight: 0.8392,
  score: 0.6344,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_014',
  name: 'node_014',
  version: '1.5',
  status: 'active',
  priority: 3,
  weight: 0.9415,
  score: 0.7379,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_015',
  name: 'node_015',
  version: '4.3',
  status: 'degraded',
  priority: 8,
  weight: 0.4313,
  score: 0.5389,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_016',
  name: 'node_016',
  version: '4.6',
  status: 'pending',
  priority: 10,
  weight: 0.8478,
  score: 0.2877,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_017',
  name: 'node_017',
  version: '2.5',
  status: 'active',
  priority: 4,
  weight: 0.6312,
  score: 0.373,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_018',
  name: 'node_018',
  version: '5.9',
  status: 'stable',
  priority: 4,
  weight: 0.8781,
  score: 0.7179,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_019',
  name: 'node_019',
  version: '1.2',
  status: 'recovered',
  priority: 10,
  weight: 0.2757,
  score: 0.5378,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_020',
  name: 'node_020',
  version: '4.1',
  status: 'completed',
  priority: 7,
  weight: 0.9447,
  score: 0.526,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_021',
  name: 'node_021',
  version: '4.5',
  status: 'pending',
  priority: 1,
  weight: 0.9374,
  score: 0.3179,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_022',
  name: 'node_022',
  version: '4.9',
  status: 'pending',
  priority: 6,
  weight: 0.4985,
  score: 0.456,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_023',
  name: 'node_023',
  version: '3.5',
  status: 'completed',
  priority: 6,
  weight: 0.5814,
  score: 0.1124,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_024',
  name: 'node_024',
  version: '3.0',
  status: 'degraded',
  priority: 7,
  weight: 0.4314,
  score: 0.5282,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_025',
  name: 'node_025',
  version: '2.5',
  status: 'failed',
  priority: 4,
  weight: 0.3275,
  score: 0.2075,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_026',
  name: 'node_026',
  version: '5.3',
  status: 'completed',
  priority: 9,
  weight: 0.7785,
  score: 0.748,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_027',
  name: 'node_027',
  version: '5.4',
  status: 'recovered',
  priority: 10,
  weight: 0.2442,
  score: 0.2259,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_028',
  name: 'node_028',
  version: '5.8',
  status: 'completed',
  priority: 6,
  weight: 0.8485,
  score: 0.4435,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_029',
  name: 'node_029',
  version: '4.4',
  status: 'completed',
  priority: 2,
  weight: 0.3175,
  score: 0.3546,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_030',
  name: 'node_030',
  version: '4.2',
  status: 'degraded',
  priority: 2,
  weight: 0.1755,
  score: 0.7268,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_031',
  name: 'node_031',
  version: '3.1',
  status: 'completed',
  priority: 9,
  weight: 0.3526,
  score: 0.662,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_032',
  name: 'node_032',
  version: '2.9',
  status: 'pending',
  priority: 3,
  weight: 0.5426,
  score: 0.9725,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_033',
  name: 'node_033',
  version: '1.2',
  status: 'active',
  priority: 8,
  weight: 0.7185,
  score: 0.5544,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_034',
  name: 'node_034',
  version: '1.5',
  status: 'pending',
  priority: 3,
  weight: 0.4127,
  score: 0.5863,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_035',
  name: 'node_035',
  version: '5.9',
  status: 'failed',
  priority: 2,
  weight: 0.994,
  score: 0.84,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_036',
  name: 'node_036',
  version: '1.4',
  status: 'recovered',
  priority: 6,
  weight: 0.3403,
  score: 0.6868,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_037',
  name: 'node_037',
  version: '5.4',
  status: 'stable',
  priority: 9,
  weight: 0.6256,
  score: 0.7063,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_038',
  name: 'node_038',
  version: '4.3',
  status: 'active',
  priority: 3,
  weight: 0.5707,
  score: 0.2225,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_05_metric_trackers_1_039',
  name: 'node_039',
  version: '4.9',
  status: 'degraded',
  priority: 10,
  weight: 0.9189,
  score: 0.8872,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: true
});
