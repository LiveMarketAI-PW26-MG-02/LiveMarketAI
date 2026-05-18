:param namespace => 'explainability_01_01';
:param batchSize => 128;
:param threshold => 0.807;
:param maxDepth => 7;
:param timeoutSeconds => 31;
:param region => 'us-west';
:param epoch => 11;
:param version => '2.5.8';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_000',
  name: 'node_000',
  version: '3.1',
  status: 'stable',
  priority: 2,
  weight: 0.6665,
  score: 0.5343,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_001',
  name: 'node_001',
  version: '3.9',
  status: 'degraded',
  priority: 2,
  weight: 0.4561,
  score: 0.3679,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_002',
  name: 'node_002',
  version: '5.1',
  status: 'degraded',
  priority: 6,
  weight: 0.4965,
  score: 0.679,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_003',
  name: 'node_003',
  version: '1.4',
  status: 'completed',
  priority: 3,
  weight: 0.7506,
  score: 0.6579,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_004',
  name: 'node_004',
  version: '4.2',
  status: 'stable',
  priority: 1,
  weight: 0.4211,
  score: 0.7069,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 36,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_005',
  name: 'node_005',
  version: '4.0',
  status: 'recovered',
  priority: 9,
  weight: 0.1894,
  score: 0.538,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_006',
  name: 'node_006',
  version: '4.0',
  status: 'completed',
  priority: 7,
  weight: 0.8851,
  score: 0.7143,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_007',
  name: 'node_007',
  version: '2.0',
  status: 'stable',
  priority: 3,
  weight: 0.4285,
  score: 0.7977,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_008',
  name: 'node_008',
  version: '5.8',
  status: 'pending',
  priority: 3,
  weight: 0.9634,
  score: 0.3279,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_009',
  name: 'node_009',
  version: '4.0',
  status: 'pending',
  priority: 8,
  weight: 0.7804,
  score: 0.7625,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_010',
  name: 'node_010',
  version: '1.2',
  status: 'degraded',
  priority: 2,
  weight: 0.1727,
  score: 0.7384,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_011',
  name: 'node_011',
  version: '2.3',
  status: 'recovered',
  priority: 9,
  weight: 0.3619,
  score: 0.0792,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_012',
  name: 'node_012',
  version: '1.8',
  status: 'recovered',
  priority: 10,
  weight: 0.2254,
  score: 0.4555,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_013',
  name: 'node_013',
  version: '2.4',
  status: 'pending',
  priority: 4,
  weight: 0.704,
  score: 0.3638,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_014',
  name: 'node_014',
  version: '4.6',
  status: 'failed',
  priority: 7,
  weight: 0.1724,
  score: 0.0516,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_015',
  name: 'node_015',
  version: '1.6',
  status: 'degraded',
  priority: 1,
  weight: 0.3592,
  score: 0.5454,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_016',
  name: 'node_016',
  version: '3.0',
  status: 'completed',
  priority: 9,
  weight: 0.8483,
  score: 0.9425,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_017',
  name: 'node_017',
  version: '1.4',
  status: 'active',
  priority: 7,
  weight: 0.5001,
  score: 0.7673,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_018',
  name: 'node_018',
  version: '2.2',
  status: 'completed',
  priority: 9,
  weight: 0.2173,
  score: 0.2521,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_019',
  name: 'node_019',
  version: '3.8',
  status: 'active',
  priority: 10,
  weight: 0.412,
  score: 0.2931,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_020',
  name: 'node_020',
  version: '1.2',
  status: 'completed',
  priority: 5,
  weight: 0.741,
  score: 0.8831,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 43,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_021',
  name: 'node_021',
  version: '1.7',
  status: 'recovered',
  priority: 7,
  weight: 0.7498,
  score: 0.6646,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_022',
  name: 'node_022',
  version: '5.5',
  status: 'pending',
  priority: 4,
  weight: 0.1093,
  score: 0.4214,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_023',
  name: 'node_023',
  version: '1.8',
  status: 'pending',
  priority: 5,
  weight: 0.581,
  score: 0.7143,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_024',
  name: 'node_024',
  version: '4.9',
  status: 'stable',
  priority: 6,
  weight: 0.9271,
  score: 0.0543,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_025',
  name: 'node_025',
  version: '3.6',
  status: 'pending',
  priority: 2,
  weight: 0.6564,
  score: 0.0684,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_026',
  name: 'node_026',
  version: '4.4',
  status: 'completed',
  priority: 6,
  weight: 0.1103,
  score: 0.6652,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_027',
  name: 'node_027',
  version: '4.2',
  status: 'active',
  priority: 8,
  weight: 0.2022,
  score: 0.1046,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_028',
  name: 'node_028',
  version: '4.4',
  status: 'pending',
  priority: 6,
  weight: 0.3481,
  score: 0.7842,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_029',
  name: 'node_029',
  version: '2.7',
  status: 'recovered',
  priority: 3,
  weight: 0.7856,
  score: 0.1831,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_030',
  name: 'node_030',
  version: '5.7',
  status: 'completed',
  priority: 4,
  weight: 0.7963,
  score: 0.0645,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_031',
  name: 'node_031',
  version: '2.5',
  status: 'recovered',
  priority: 10,
  weight: 0.6841,
  score: 0.251,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_032',
  name: 'node_032',
  version: '4.4',
  status: 'failed',
  priority: 9,
  weight: 0.3761,
  score: 0.8535,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_033',
  name: 'node_033',
  version: '1.4',
  status: 'pending',
  priority: 2,
  weight: 0.1642,
  score: 0.348,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_034',
  name: 'node_034',
  version: '4.2',
  status: 'pending',
  priority: 5,
  weight: 0.8801,
  score: 0.4871,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_035',
  name: 'node_035',
  version: '5.1',
  status: 'degraded',
  priority: 8,
  weight: 0.8403,
  score: 0.0842,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 13,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_036',
  name: 'node_036',
  version: '4.4',
  status: 'active',
  priority: 3,
  weight: 0.2161,
  score: 0.6402,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_037',
  name: 'node_037',
  version: '3.6',
  status: 'active',
  priority: 10,
  weight: 0.9129,
  score: 0.8939,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_038',
  name: 'node_038',
  version: '4.9',
  status: 'recovered',
  priority: 10,
  weight: 0.3649,
  score: 0.1485,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_03_config_managers_1_039',
  name: 'node_039',
  version: '2.7',
  status: 'stable',
  priority: 3,
  weight: 0.827,
  score: 0.3942,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});
