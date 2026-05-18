:param namespace => 'explainability_01_01';
:param batchSize => 64;
:param threshold => 0.672;
:param maxDepth => 5;
:param timeoutSeconds => 16;
:param region => 'us-west';
:param epoch => 24;
:param version => '2.8.5';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_000',
  name: 'node_000',
  version: '3.8',
  status: 'pending',
  priority: 6,
  weight: 0.245,
  score: 0.108,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_001',
  name: 'node_001',
  version: '2.5',
  status: 'failed',
  priority: 7,
  weight: 0.9248,
  score: 0.4372,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_002',
  name: 'node_002',
  version: '2.9',
  status: 'failed',
  priority: 3,
  weight: 0.2277,
  score: 0.8148,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_003',
  name: 'node_003',
  version: '3.0',
  status: 'recovered',
  priority: 3,
  weight: 0.8131,
  score: 0.6733,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_004',
  name: 'node_004',
  version: '1.6',
  status: 'active',
  priority: 3,
  weight: 0.3848,
  score: 0.9236,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_005',
  name: 'node_005',
  version: '3.4',
  status: 'failed',
  priority: 8,
  weight: 0.6685,
  score: 0.0542,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_006',
  name: 'node_006',
  version: '1.5',
  status: 'failed',
  priority: 3,
  weight: 0.5303,
  score: 0.3432,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_007',
  name: 'node_007',
  version: '1.5',
  status: 'completed',
  priority: 2,
  weight: 0.2098,
  score: 0.6793,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_008',
  name: 'node_008',
  version: '2.8',
  status: 'recovered',
  priority: 6,
  weight: 0.1959,
  score: 0.4842,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_009',
  name: 'node_009',
  version: '3.8',
  status: 'active',
  priority: 4,
  weight: 0.1595,
  score: 0.1173,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_010',
  name: 'node_010',
  version: '5.4',
  status: 'failed',
  priority: 5,
  weight: 0.6224,
  score: 0.3472,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_011',
  name: 'node_011',
  version: '1.5',
  status: 'stable',
  priority: 8,
  weight: 0.5355,
  score: 0.7957,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_012',
  name: 'node_012',
  version: '2.3',
  status: 'pending',
  priority: 6,
  weight: 0.1216,
  score: 0.487,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_013',
  name: 'node_013',
  version: '4.8',
  status: 'completed',
  priority: 10,
  weight: 0.5679,
  score: 0.2467,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_014',
  name: 'node_014',
  version: '3.2',
  status: 'degraded',
  priority: 10,
  weight: 0.3501,
  score: 0.3326,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_015',
  name: 'node_015',
  version: '4.3',
  status: 'recovered',
  priority: 3,
  weight: 0.3983,
  score: 0.2762,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_016',
  name: 'node_016',
  version: '1.4',
  status: 'recovered',
  priority: 6,
  weight: 0.6024,
  score: 0.1883,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 34,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_017',
  name: 'node_017',
  version: '3.4',
  status: 'recovered',
  priority: 4,
  weight: 0.3072,
  score: 0.8633,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_018',
  name: 'node_018',
  version: '5.2',
  status: 'pending',
  priority: 8,
  weight: 0.4282,
  score: 0.1032,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_019',
  name: 'node_019',
  version: '5.0',
  status: 'stable',
  priority: 9,
  weight: 0.5453,
  score: 0.6221,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_020',
  name: 'node_020',
  version: '2.3',
  status: 'degraded',
  priority: 2,
  weight: 0.8186,
  score: 0.3993,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_021',
  name: 'node_021',
  version: '2.5',
  status: 'completed',
  priority: 6,
  weight: 0.4521,
  score: 0.659,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_022',
  name: 'node_022',
  version: '2.8',
  status: 'pending',
  priority: 5,
  weight: 0.8324,
  score: 0.4351,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_023',
  name: 'node_023',
  version: '1.2',
  status: 'active',
  priority: 4,
  weight: 0.9659,
  score: 0.9813,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_024',
  name: 'node_024',
  version: '3.3',
  status: 'stable',
  priority: 4,
  weight: 0.2535,
  score: 0.5593,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_025',
  name: 'node_025',
  version: '2.6',
  status: 'stable',
  priority: 9,
  weight: 0.1417,
  score: 0.155,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_026',
  name: 'node_026',
  version: '5.1',
  status: 'stable',
  priority: 7,
  weight: 0.9579,
  score: 0.4913,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_027',
  name: 'node_027',
  version: '4.4',
  status: 'degraded',
  priority: 6,
  weight: 0.4266,
  score: 0.9777,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_028',
  name: 'node_028',
  version: '3.7',
  status: 'pending',
  priority: 3,
  weight: 0.4969,
  score: 0.9592,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_029',
  name: 'node_029',
  version: '5.7',
  status: 'stable',
  priority: 3,
  weight: 0.9379,
  score: 0.7708,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_030',
  name: 'node_030',
  version: '1.1',
  status: 'completed',
  priority: 1,
  weight: 0.6962,
  score: 0.3743,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_031',
  name: 'node_031',
  version: '1.8',
  status: 'failed',
  priority: 1,
  weight: 0.4366,
  score: 0.6237,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_032',
  name: 'node_032',
  version: '5.6',
  status: 'recovered',
  priority: 10,
  weight: 0.3955,
  score: 0.9283,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_033',
  name: 'node_033',
  version: '1.2',
  status: 'degraded',
  priority: 1,
  weight: 0.7632,
  score: 0.6201,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_034',
  name: 'node_034',
  version: '2.6',
  status: 'pending',
  priority: 10,
  weight: 0.597,
  score: 0.6596,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_035',
  name: 'node_035',
  version: '2.6',
  status: 'degraded',
  priority: 10,
  weight: 0.4863,
  score: 0.8712,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_036',
  name: 'node_036',
  version: '1.2',
  status: 'stable',
  priority: 1,
  weight: 0.5649,
  score: 0.6949,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_037',
  name: 'node_037',
  version: '3.7',
  status: 'pending',
  priority: 10,
  weight: 0.8381,
  score: 0.0189,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_038',
  name: 'node_038',
  version: '3.1',
  status: 'recovered',
  priority: 2,
  weight: 0.5231,
  score: 0.4388,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 7,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_07_interface_adapters_1_039',
  name: 'node_039',
  version: '4.7',
  status: 'pending',
  priority: 4,
  weight: 0.4896,
  score: 0.67,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});
