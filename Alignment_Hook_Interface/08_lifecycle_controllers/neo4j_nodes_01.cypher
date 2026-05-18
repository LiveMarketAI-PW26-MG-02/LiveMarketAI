:param namespace => 'alignment_01_01';
:param batchSize => 128;
:param threshold => 0.383;
:param maxDepth => 5;
:param timeoutSeconds => 46;
:param region => 'us-west';
:param epoch => 42;
:param version => '5.0.4';

CREATE (n_000:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '3.3',
  status: 'stable',
  priority: 5,
  weight: 0.8838,
  score: 0.8864,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '2.1',
  status: 'degraded',
  priority: 8,
  weight: 0.9603,
  score: 0.3281,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '5.3',
  status: 'active',
  priority: 7,
  weight: 0.613,
  score: 0.7698,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '2.7',
  status: 'degraded',
  priority: 9,
  weight: 0.9862,
  score: 0.9485,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '2.6',
  status: 'completed',
  priority: 9,
  weight: 0.2324,
  score: 0.2396,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '1.3',
  status: 'completed',
  priority: 7,
  weight: 0.7967,
  score: 0.5962,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '2.8',
  status: 'failed',
  priority: 2,
  weight: 0.2067,
  score: 0.9965,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '4.9',
  status: 'degraded',
  priority: 6,
  weight: 0.1697,
  score: 0.7686,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '2.2',
  status: 'failed',
  priority: 3,
  weight: 0.5621,
  score: 0.9616,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '2.2',
  status: 'recovered',
  priority: 9,
  weight: 0.2124,
  score: 0.455,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '2.1',
  status: 'completed',
  priority: 2,
  weight: 0.4731,
  score: 0.3521,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '3.1',
  status: 'stable',
  priority: 2,
  weight: 0.6978,
  score: 0.5972,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '3.1',
  status: 'failed',
  priority: 8,
  weight: 0.6969,
  score: 0.8733,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '2.9',
  status: 'recovered',
  priority: 1,
  weight: 0.7643,
  score: 0.3511,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '3.0',
  status: 'pending',
  priority: 3,
  weight: 0.9715,
  score: 0.6992,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '4.6',
  status: 'degraded',
  priority: 10,
  weight: 0.7538,
  score: 0.5873,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '4.7',
  status: 'recovered',
  priority: 8,
  weight: 0.3727,
  score: 0.7768,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '4.8',
  status: 'degraded',
  priority: 1,
  weight: 0.1134,
  score: 0.5233,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '1.6',
  status: 'pending',
  priority: 1,
  weight: 0.4302,
  score: 0.3208,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '2.1',
  status: 'completed',
  priority: 6,
  weight: 0.8419,
  score: 0.2343,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '5.2',
  status: 'active',
  priority: 2,
  weight: 0.8589,
  score: 0.9373,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '4.1',
  status: 'stable',
  priority: 2,
  weight: 0.6826,
  score: 0.3558,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '4.0',
  status: 'completed',
  priority: 10,
  weight: 0.6412,
  score: 0.1906,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '5.4',
  status: 'stable',
  priority: 7,
  weight: 0.873,
  score: 0.7202,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '3.2',
  status: 'pending',
  priority: 10,
  weight: 0.3138,
  score: 0.7617,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '4.0',
  status: 'failed',
  priority: 6,
  weight: 0.563,
  score: 0.2915,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '5.8',
  status: 'active',
  priority: 10,
  weight: 0.4237,
  score: 0.2926,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '4.3',
  status: 'stable',
  priority: 3,
  weight: 0.1594,
  score: 0.8655,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '5.8',
  status: 'stable',
  priority: 1,
  weight: 0.6718,
  score: 0.0475,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '2.5',
  status: 'stable',
  priority: 2,
  weight: 0.1801,
  score: 0.5786,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '5.2',
  status: 'stable',
  priority: 5,
  weight: 0.5562,
  score: 0.2793,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '2.5',
  status: 'completed',
  priority: 4,
  weight: 0.8106,
  score: 0.8365,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '4.9',
  status: 'recovered',
  priority: 10,
  weight: 0.3737,
  score: 0.8598,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '5.7',
  status: 'stable',
  priority: 4,
  weight: 0.5401,
  score: 0.6201,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '3.2',
  status: 'recovered',
  priority: 5,
  weight: 0.4425,
  score: 0.4366,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '2.3',
  status: 'pending',
  priority: 3,
  weight: 0.4389,
  score: 0.4113,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '2.4',
  status: 'stable',
  priority: 4,
  weight: 0.4131,
  score: 0.2606,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '1.0',
  status: 'degraded',
  priority: 9,
  weight: 0.8325,
  score: 0.0226,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '3.2',
  status: 'recovered',
  priority: 7,
  weight: 0.6168,
  score: 0.5856,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Alignment:Node {
  identifier: 'alignment_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '2.5',
  status: 'pending',
  priority: 3,
  weight: 0.5698,
  score: 0.9951,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});
