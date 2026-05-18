:param namespace => 'compression_01_01';
:param batchSize => 512;
:param threshold => 0.567;
:param maxDepth => 5;
:param timeoutSeconds => 117;
:param region => 'us-west';
:param epoch => 63;
:param version => '5.3.7';

CREATE (n_000:Compression:Node {
  identifier: 'compression_02_state_handlers_1_000',
  name: 'node_000',
  version: '4.3',
  status: 'completed',
  priority: 4,
  weight: 0.3272,
  score: 0.3953,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_02_state_handlers_1_001',
  name: 'node_001',
  version: '5.5',
  status: 'completed',
  priority: 7,
  weight: 0.8617,
  score: 0.1812,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_02_state_handlers_1_002',
  name: 'node_002',
  version: '1.4',
  status: 'degraded',
  priority: 6,
  weight: 0.4653,
  score: 0.2323,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_02_state_handlers_1_003',
  name: 'node_003',
  version: '4.7',
  status: 'recovered',
  priority: 7,
  weight: 0.7046,
  score: 0.7177,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_02_state_handlers_1_004',
  name: 'node_004',
  version: '1.0',
  status: 'degraded',
  priority: 3,
  weight: 0.7738,
  score: 0.2834,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_02_state_handlers_1_005',
  name: 'node_005',
  version: '3.3',
  status: 'stable',
  priority: 8,
  weight: 0.1071,
  score: 0.9791,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_02_state_handlers_1_006',
  name: 'node_006',
  version: '3.8',
  status: 'completed',
  priority: 1,
  weight: 0.55,
  score: 0.4978,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_02_state_handlers_1_007',
  name: 'node_007',
  version: '2.4',
  status: 'stable',
  priority: 2,
  weight: 0.9785,
  score: 0.7396,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_02_state_handlers_1_008',
  name: 'node_008',
  version: '3.0',
  status: 'active',
  priority: 9,
  weight: 0.5537,
  score: 0.9138,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_02_state_handlers_1_009',
  name: 'node_009',
  version: '2.2',
  status: 'completed',
  priority: 10,
  weight: 0.7897,
  score: 0.2618,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_02_state_handlers_1_010',
  name: 'node_010',
  version: '3.4',
  status: 'recovered',
  priority: 7,
  weight: 0.2995,
  score: 0.9188,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_02_state_handlers_1_011',
  name: 'node_011',
  version: '5.9',
  status: 'active',
  priority: 4,
  weight: 0.1286,
  score: 0.3904,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_02_state_handlers_1_012',
  name: 'node_012',
  version: '1.3',
  status: 'stable',
  priority: 8,
  weight: 0.8327,
  score: 0.2159,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_02_state_handlers_1_013',
  name: 'node_013',
  version: '2.5',
  status: 'pending',
  priority: 8,
  weight: 0.2813,
  score: 0.9731,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_02_state_handlers_1_014',
  name: 'node_014',
  version: '5.1',
  status: 'stable',
  priority: 7,
  weight: 0.461,
  score: 0.0478,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_02_state_handlers_1_015',
  name: 'node_015',
  version: '4.3',
  status: 'degraded',
  priority: 7,
  weight: 0.5774,
  score: 0.3119,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_02_state_handlers_1_016',
  name: 'node_016',
  version: '2.3',
  status: 'degraded',
  priority: 8,
  weight: 0.4527,
  score: 0.9453,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_02_state_handlers_1_017',
  name: 'node_017',
  version: '3.9',
  status: 'failed',
  priority: 2,
  weight: 0.8579,
  score: 0.0346,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_02_state_handlers_1_018',
  name: 'node_018',
  version: '5.9',
  status: 'completed',
  priority: 8,
  weight: 0.2178,
  score: 0.2206,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_02_state_handlers_1_019',
  name: 'node_019',
  version: '4.4',
  status: 'failed',
  priority: 8,
  weight: 0.831,
  score: 0.9349,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 8,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_02_state_handlers_1_020',
  name: 'node_020',
  version: '4.9',
  status: 'failed',
  priority: 7,
  weight: 0.2312,
  score: 0.1862,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_02_state_handlers_1_021',
  name: 'node_021',
  version: '5.9',
  status: 'recovered',
  priority: 9,
  weight: 0.9855,
  score: 0.1101,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_02_state_handlers_1_022',
  name: 'node_022',
  version: '3.4',
  status: 'active',
  priority: 3,
  weight: 0.6126,
  score: 0.2735,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_02_state_handlers_1_023',
  name: 'node_023',
  version: '1.4',
  status: 'completed',
  priority: 1,
  weight: 0.9471,
  score: 0.757,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_02_state_handlers_1_024',
  name: 'node_024',
  version: '2.0',
  status: 'pending',
  priority: 5,
  weight: 0.301,
  score: 0.9361,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_02_state_handlers_1_025',
  name: 'node_025',
  version: '1.9',
  status: 'degraded',
  priority: 8,
  weight: 0.7471,
  score: 0.3996,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_02_state_handlers_1_026',
  name: 'node_026',
  version: '2.3',
  status: 'recovered',
  priority: 5,
  weight: 0.6521,
  score: 0.6464,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_02_state_handlers_1_027',
  name: 'node_027',
  version: '5.0',
  status: 'pending',
  priority: 7,
  weight: 0.7502,
  score: 0.2437,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_02_state_handlers_1_028',
  name: 'node_028',
  version: '2.6',
  status: 'recovered',
  priority: 6,
  weight: 0.9288,
  score: 0.1901,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_02_state_handlers_1_029',
  name: 'node_029',
  version: '4.9',
  status: 'completed',
  priority: 7,
  weight: 0.9648,
  score: 0.0805,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_02_state_handlers_1_030',
  name: 'node_030',
  version: '3.2',
  status: 'degraded',
  priority: 3,
  weight: 0.3494,
  score: 0.2517,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_02_state_handlers_1_031',
  name: 'node_031',
  version: '3.7',
  status: 'pending',
  priority: 4,
  weight: 0.8751,
  score: 0.5337,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_02_state_handlers_1_032',
  name: 'node_032',
  version: '4.1',
  status: 'completed',
  priority: 9,
  weight: 0.6088,
  score: 0.5198,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_02_state_handlers_1_033',
  name: 'node_033',
  version: '3.4',
  status: 'recovered',
  priority: 2,
  weight: 0.8745,
  score: 0.3428,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_02_state_handlers_1_034',
  name: 'node_034',
  version: '4.7',
  status: 'recovered',
  priority: 2,
  weight: 0.6672,
  score: 0.7003,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_02_state_handlers_1_035',
  name: 'node_035',
  version: '5.4',
  status: 'degraded',
  priority: 1,
  weight: 0.1219,
  score: 0.2308,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 79,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_02_state_handlers_1_036',
  name: 'node_036',
  version: '4.8',
  status: 'pending',
  priority: 1,
  weight: 0.4367,
  score: 0.9421,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_02_state_handlers_1_037',
  name: 'node_037',
  version: '5.7',
  status: 'completed',
  priority: 2,
  weight: 0.1894,
  score: 0.825,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_02_state_handlers_1_038',
  name: 'node_038',
  version: '2.3',
  status: 'recovered',
  priority: 1,
  weight: 0.2534,
  score: 0.528,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_02_state_handlers_1_039',
  name: 'node_039',
  version: '1.9',
  status: 'pending',
  priority: 8,
  weight: 0.3418,
  score: 0.2457,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 86,
  createdAt: datetime(),
  active: false
});
