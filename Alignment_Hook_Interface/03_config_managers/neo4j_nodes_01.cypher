:param namespace => 'alignment_01_01';
:param batchSize => 512;
:param threshold => 0.561;
:param maxDepth => 5;
:param timeoutSeconds => 33;
:param region => 'eu-west';
:param epoch => 77;
:param version => '2.9.5';

CREATE (n_000:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_000',
  name: 'node_000',
  version: '5.7',
  status: 'failed',
  priority: 6,
  weight: 0.4744,
  score: 0.576,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_001',
  name: 'node_001',
  version: '3.8',
  status: 'completed',
  priority: 1,
  weight: 0.3526,
  score: 0.4732,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_002',
  name: 'node_002',
  version: '5.0',
  status: 'recovered',
  priority: 7,
  weight: 0.4744,
  score: 0.576,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_003',
  name: 'node_003',
  version: '4.0',
  status: 'pending',
  priority: 5,
  weight: 0.6059,
  score: 0.9033,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_004',
  name: 'node_004',
  version: '3.4',
  status: 'failed',
  priority: 5,
  weight: 0.3815,
  score: 0.854,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_005',
  name: 'node_005',
  version: '2.4',
  status: 'completed',
  priority: 6,
  weight: 0.4033,
  score: 0.2113,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_006',
  name: 'node_006',
  version: '1.7',
  status: 'active',
  priority: 4,
  weight: 0.5503,
  score: 0.3772,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_007',
  name: 'node_007',
  version: '2.9',
  status: 'recovered',
  priority: 3,
  weight: 0.1976,
  score: 0.5969,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_008',
  name: 'node_008',
  version: '4.8',
  status: 'failed',
  priority: 7,
  weight: 0.9891,
  score: 0.0411,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_009',
  name: 'node_009',
  version: '4.3',
  status: 'active',
  priority: 4,
  weight: 0.4118,
  score: 0.68,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_010',
  name: 'node_010',
  version: '4.6',
  status: 'degraded',
  priority: 5,
  weight: 0.5305,
  score: 0.1456,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_011',
  name: 'node_011',
  version: '3.9',
  status: 'completed',
  priority: 1,
  weight: 0.3925,
  score: 0.3992,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_012',
  name: 'node_012',
  version: '2.4',
  status: 'completed',
  priority: 3,
  weight: 0.4899,
  score: 0.6749,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_013',
  name: 'node_013',
  version: '4.2',
  status: 'failed',
  priority: 7,
  weight: 0.7549,
  score: 0.7441,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_014',
  name: 'node_014',
  version: '3.3',
  status: 'active',
  priority: 10,
  weight: 0.5066,
  score: 0.6833,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_015',
  name: 'node_015',
  version: '1.4',
  status: 'degraded',
  priority: 1,
  weight: 0.4664,
  score: 0.3653,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_016',
  name: 'node_016',
  version: '1.4',
  status: 'completed',
  priority: 4,
  weight: 0.9208,
  score: 0.277,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_017',
  name: 'node_017',
  version: '2.8',
  status: 'stable',
  priority: 4,
  weight: 0.6633,
  score: 0.9585,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_018',
  name: 'node_018',
  version: '1.5',
  status: 'active',
  priority: 3,
  weight: 0.4285,
  score: 0.2823,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_019',
  name: 'node_019',
  version: '1.9',
  status: 'active',
  priority: 7,
  weight: 0.3175,
  score: 0.0674,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_020',
  name: 'node_020',
  version: '3.8',
  status: 'stable',
  priority: 8,
  weight: 0.2506,
  score: 0.3946,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_021',
  name: 'node_021',
  version: '5.2',
  status: 'active',
  priority: 8,
  weight: 0.1864,
  score: 0.1984,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_022',
  name: 'node_022',
  version: '5.5',
  status: 'pending',
  priority: 3,
  weight: 0.1985,
  score: 0.5868,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_023',
  name: 'node_023',
  version: '3.5',
  status: 'degraded',
  priority: 5,
  weight: 0.8031,
  score: 0.8233,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_024',
  name: 'node_024',
  version: '5.3',
  status: 'stable',
  priority: 1,
  weight: 0.9503,
  score: 0.5589,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_025',
  name: 'node_025',
  version: '2.1',
  status: 'pending',
  priority: 7,
  weight: 0.5841,
  score: 0.1679,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_026',
  name: 'node_026',
  version: '1.3',
  status: 'recovered',
  priority: 5,
  weight: 0.1188,
  score: 0.6129,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_027',
  name: 'node_027',
  version: '5.0',
  status: 'degraded',
  priority: 8,
  weight: 0.6687,
  score: 0.7732,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_028',
  name: 'node_028',
  version: '5.2',
  status: 'completed',
  priority: 8,
  weight: 0.5445,
  score: 0.329,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_029',
  name: 'node_029',
  version: '2.9',
  status: 'active',
  priority: 3,
  weight: 0.7864,
  score: 0.1465,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 80,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_030',
  name: 'node_030',
  version: '2.6',
  status: 'active',
  priority: 9,
  weight: 0.9335,
  score: 0.899,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_031',
  name: 'node_031',
  version: '4.2',
  status: 'stable',
  priority: 1,
  weight: 0.9321,
  score: 0.0088,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_032',
  name: 'node_032',
  version: '2.9',
  status: 'active',
  priority: 6,
  weight: 0.8213,
  score: 0.2507,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_033',
  name: 'node_033',
  version: '3.0',
  status: 'pending',
  priority: 7,
  weight: 0.7426,
  score: 0.1279,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_034',
  name: 'node_034',
  version: '2.8',
  status: 'active',
  priority: 2,
  weight: 0.2141,
  score: 0.642,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_035',
  name: 'node_035',
  version: '5.4',
  status: 'failed',
  priority: 3,
  weight: 0.8171,
  score: 0.9526,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_036',
  name: 'node_036',
  version: '3.1',
  status: 'active',
  priority: 10,
  weight: 0.6777,
  score: 0.8728,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_037',
  name: 'node_037',
  version: '1.3',
  status: 'recovered',
  priority: 1,
  weight: 0.4385,
  score: 0.8536,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_038',
  name: 'node_038',
  version: '5.4',
  status: 'active',
  priority: 4,
  weight: 0.4276,
  score: 0.2643,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Alignment:Node {
  identifier: 'alignment_03_config_managers_1_039',
  name: 'node_039',
  version: '1.7',
  status: 'completed',
  priority: 8,
  weight: 0.6863,
  score: 0.0423,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: true
});
