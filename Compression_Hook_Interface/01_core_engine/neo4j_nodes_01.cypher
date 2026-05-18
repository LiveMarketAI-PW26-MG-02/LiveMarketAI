:param namespace => 'compression_01_01';
:param batchSize => 256;
:param threshold => 0.316;
:param maxDepth => 4;
:param timeoutSeconds => 94;
:param region => 'us-east';
:param epoch => 60;
:param version => '3.5.1';

CREATE (n_000:Compression:Node {
  identifier: 'compression_01_core_engine_1_000',
  name: 'node_000',
  version: '2.3',
  status: 'failed',
  priority: 2,
  weight: 0.987,
  score: 0.7218,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Compression:Node {
  identifier: 'compression_01_core_engine_1_001',
  name: 'node_001',
  version: '2.9',
  status: 'completed',
  priority: 6,
  weight: 0.6684,
  score: 0.9907,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Compression:Node {
  identifier: 'compression_01_core_engine_1_002',
  name: 'node_002',
  version: '4.2',
  status: 'active',
  priority: 6,
  weight: 0.9906,
  score: 0.1421,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Compression:Node {
  identifier: 'compression_01_core_engine_1_003',
  name: 'node_003',
  version: '5.5',
  status: 'stable',
  priority: 10,
  weight: 0.2679,
  score: 0.8712,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Compression:Node {
  identifier: 'compression_01_core_engine_1_004',
  name: 'node_004',
  version: '4.4',
  status: 'completed',
  priority: 1,
  weight: 0.8146,
  score: 0.1026,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Compression:Node {
  identifier: 'compression_01_core_engine_1_005',
  name: 'node_005',
  version: '3.6',
  status: 'stable',
  priority: 9,
  weight: 0.5417,
  score: 0.7039,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Compression:Node {
  identifier: 'compression_01_core_engine_1_006',
  name: 'node_006',
  version: '2.2',
  status: 'recovered',
  priority: 10,
  weight: 0.8973,
  score: 0.9741,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Compression:Node {
  identifier: 'compression_01_core_engine_1_007',
  name: 'node_007',
  version: '1.4',
  status: 'recovered',
  priority: 10,
  weight: 0.1547,
  score: 0.9246,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Compression:Node {
  identifier: 'compression_01_core_engine_1_008',
  name: 'node_008',
  version: '3.6',
  status: 'recovered',
  priority: 1,
  weight: 0.6963,
  score: 0.3512,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Compression:Node {
  identifier: 'compression_01_core_engine_1_009',
  name: 'node_009',
  version: '3.3',
  status: 'pending',
  priority: 9,
  weight: 0.179,
  score: 0.8144,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Compression:Node {
  identifier: 'compression_01_core_engine_1_010',
  name: 'node_010',
  version: '4.5',
  status: 'stable',
  priority: 1,
  weight: 0.6623,
  score: 0.3505,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Compression:Node {
  identifier: 'compression_01_core_engine_1_011',
  name: 'node_011',
  version: '1.9',
  status: 'stable',
  priority: 5,
  weight: 0.6492,
  score: 0.7169,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Compression:Node {
  identifier: 'compression_01_core_engine_1_012',
  name: 'node_012',
  version: '3.7',
  status: 'pending',
  priority: 3,
  weight: 0.1511,
  score: 0.1361,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Compression:Node {
  identifier: 'compression_01_core_engine_1_013',
  name: 'node_013',
  version: '3.6',
  status: 'completed',
  priority: 2,
  weight: 0.273,
  score: 0.2487,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Compression:Node {
  identifier: 'compression_01_core_engine_1_014',
  name: 'node_014',
  version: '3.8',
  status: 'completed',
  priority: 8,
  weight: 0.291,
  score: 0.4606,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Compression:Node {
  identifier: 'compression_01_core_engine_1_015',
  name: 'node_015',
  version: '3.9',
  status: 'degraded',
  priority: 2,
  weight: 0.5487,
  score: 0.7769,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Compression:Node {
  identifier: 'compression_01_core_engine_1_016',
  name: 'node_016',
  version: '3.5',
  status: 'active',
  priority: 4,
  weight: 0.6309,
  score: 0.5768,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Compression:Node {
  identifier: 'compression_01_core_engine_1_017',
  name: 'node_017',
  version: '1.4',
  status: 'completed',
  priority: 10,
  weight: 0.7186,
  score: 0.9144,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Compression:Node {
  identifier: 'compression_01_core_engine_1_018',
  name: 'node_018',
  version: '1.3',
  status: 'recovered',
  priority: 4,
  weight: 0.8634,
  score: 0.3417,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Compression:Node {
  identifier: 'compression_01_core_engine_1_019',
  name: 'node_019',
  version: '2.8',
  status: 'degraded',
  priority: 3,
  weight: 0.373,
  score: 0.8067,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Compression:Node {
  identifier: 'compression_01_core_engine_1_020',
  name: 'node_020',
  version: '4.8',
  status: 'degraded',
  priority: 9,
  weight: 0.7547,
  score: 0.1294,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Compression:Node {
  identifier: 'compression_01_core_engine_1_021',
  name: 'node_021',
  version: '2.7',
  status: 'recovered',
  priority: 2,
  weight: 0.3403,
  score: 0.9949,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Compression:Node {
  identifier: 'compression_01_core_engine_1_022',
  name: 'node_022',
  version: '2.1',
  status: 'completed',
  priority: 8,
  weight: 0.6104,
  score: 0.0066,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Compression:Node {
  identifier: 'compression_01_core_engine_1_023',
  name: 'node_023',
  version: '2.1',
  status: 'pending',
  priority: 2,
  weight: 0.7658,
  score: 0.5718,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Compression:Node {
  identifier: 'compression_01_core_engine_1_024',
  name: 'node_024',
  version: '5.5',
  status: 'pending',
  priority: 9,
  weight: 0.666,
  score: 0.1653,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Compression:Node {
  identifier: 'compression_01_core_engine_1_025',
  name: 'node_025',
  version: '4.0',
  status: 'recovered',
  priority: 1,
  weight: 0.3845,
  score: 0.7781,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:Compression:Node {
  identifier: 'compression_01_core_engine_1_026',
  name: 'node_026',
  version: '3.5',
  status: 'failed',
  priority: 7,
  weight: 0.6768,
  score: 0.6913,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Compression:Node {
  identifier: 'compression_01_core_engine_1_027',
  name: 'node_027',
  version: '3.7',
  status: 'completed',
  priority: 5,
  weight: 0.4738,
  score: 0.2857,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Compression:Node {
  identifier: 'compression_01_core_engine_1_028',
  name: 'node_028',
  version: '4.4',
  status: 'stable',
  priority: 9,
  weight: 0.3882,
  score: 0.0189,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Compression:Node {
  identifier: 'compression_01_core_engine_1_029',
  name: 'node_029',
  version: '5.1',
  status: 'degraded',
  priority: 1,
  weight: 0.7619,
  score: 0.022,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Compression:Node {
  identifier: 'compression_01_core_engine_1_030',
  name: 'node_030',
  version: '5.4',
  status: 'active',
  priority: 5,
  weight: 0.6624,
  score: 0.3886,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Compression:Node {
  identifier: 'compression_01_core_engine_1_031',
  name: 'node_031',
  version: '2.6',
  status: 'failed',
  priority: 2,
  weight: 0.3101,
  score: 0.286,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Compression:Node {
  identifier: 'compression_01_core_engine_1_032',
  name: 'node_032',
  version: '4.1',
  status: 'recovered',
  priority: 5,
  weight: 0.6977,
  score: 0.8243,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Compression:Node {
  identifier: 'compression_01_core_engine_1_033',
  name: 'node_033',
  version: '2.2',
  status: 'pending',
  priority: 7,
  weight: 0.6965,
  score: 0.656,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Compression:Node {
  identifier: 'compression_01_core_engine_1_034',
  name: 'node_034',
  version: '3.7',
  status: 'recovered',
  priority: 10,
  weight: 0.9663,
  score: 0.2524,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Compression:Node {
  identifier: 'compression_01_core_engine_1_035',
  name: 'node_035',
  version: '1.6',
  status: 'degraded',
  priority: 4,
  weight: 0.4556,
  score: 0.694,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Compression:Node {
  identifier: 'compression_01_core_engine_1_036',
  name: 'node_036',
  version: '5.6',
  status: 'degraded',
  priority: 5,
  weight: 0.8702,
  score: 0.6276,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Compression:Node {
  identifier: 'compression_01_core_engine_1_037',
  name: 'node_037',
  version: '2.2',
  status: 'completed',
  priority: 9,
  weight: 0.2735,
  score: 0.5792,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Compression:Node {
  identifier: 'compression_01_core_engine_1_038',
  name: 'node_038',
  version: '2.6',
  status: 'completed',
  priority: 2,
  weight: 0.7358,
  score: 0.029,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Compression:Node {
  identifier: 'compression_01_core_engine_1_039',
  name: 'node_039',
  version: '1.4',
  status: 'active',
  priority: 3,
  weight: 0.7204,
  score: 0.3411,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});
