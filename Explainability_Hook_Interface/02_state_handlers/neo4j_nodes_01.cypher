:param namespace => 'explainability_01_01';
:param batchSize => 128;
:param threshold => 0.456;
:param maxDepth => 7;
:param timeoutSeconds => 40;
:param region => 'eu-west';
:param epoch => 49;
:param version => '2.6.3';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_000',
  name: 'node_000',
  version: '2.1',
  status: 'recovered',
  priority: 9,
  weight: 0.8944,
  score: 0.3047,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_001',
  name: 'node_001',
  version: '3.6',
  status: 'failed',
  priority: 7,
  weight: 0.931,
  score: 0.5354,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_002',
  name: 'node_002',
  version: '5.0',
  status: 'active',
  priority: 8,
  weight: 0.6152,
  score: 0.9256,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_003',
  name: 'node_003',
  version: '2.7',
  status: 'pending',
  priority: 4,
  weight: 0.5623,
  score: 0.3418,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_004',
  name: 'node_004',
  version: '5.7',
  status: 'stable',
  priority: 1,
  weight: 0.3789,
  score: 0.9053,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_005',
  name: 'node_005',
  version: '3.3',
  status: 'recovered',
  priority: 2,
  weight: 0.3325,
  score: 0.2059,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_006',
  name: 'node_006',
  version: '4.9',
  status: 'failed',
  priority: 10,
  weight: 0.8365,
  score: 0.8382,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_007',
  name: 'node_007',
  version: '4.0',
  status: 'stable',
  priority: 6,
  weight: 0.7429,
  score: 0.8377,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_008',
  name: 'node_008',
  version: '3.3',
  status: 'degraded',
  priority: 1,
  weight: 0.7179,
  score: 0.3481,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_009',
  name: 'node_009',
  version: '5.1',
  status: 'recovered',
  priority: 3,
  weight: 0.2386,
  score: 0.0919,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_010',
  name: 'node_010',
  version: '1.7',
  status: 'completed',
  priority: 1,
  weight: 0.5528,
  score: 0.0718,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_011',
  name: 'node_011',
  version: '1.1',
  status: 'stable',
  priority: 5,
  weight: 0.3834,
  score: 0.082,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_012',
  name: 'node_012',
  version: '3.8',
  status: 'pending',
  priority: 1,
  weight: 0.9183,
  score: 0.1207,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_013',
  name: 'node_013',
  version: '3.6',
  status: 'stable',
  priority: 3,
  weight: 0.1206,
  score: 0.2665,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_014',
  name: 'node_014',
  version: '3.7',
  status: 'stable',
  priority: 2,
  weight: 0.8952,
  score: 0.2732,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 62,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_015',
  name: 'node_015',
  version: '2.1',
  status: 'pending',
  priority: 8,
  weight: 0.2503,
  score: 0.0354,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_016',
  name: 'node_016',
  version: '1.0',
  status: 'failed',
  priority: 3,
  weight: 0.5982,
  score: 0.6947,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_017',
  name: 'node_017',
  version: '1.8',
  status: 'stable',
  priority: 7,
  weight: 0.2372,
  score: 0.1073,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_018',
  name: 'node_018',
  version: '5.5',
  status: 'failed',
  priority: 2,
  weight: 0.401,
  score: 0.0588,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_019',
  name: 'node_019',
  version: '3.5',
  status: 'stable',
  priority: 7,
  weight: 0.5665,
  score: 0.1282,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_020',
  name: 'node_020',
  version: '1.2',
  status: 'pending',
  priority: 6,
  weight: 0.6213,
  score: 0.6224,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_021',
  name: 'node_021',
  version: '2.0',
  status: 'degraded',
  priority: 6,
  weight: 0.3311,
  score: 0.1079,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_022',
  name: 'node_022',
  version: '2.9',
  status: 'stable',
  priority: 3,
  weight: 0.8813,
  score: 0.8404,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_023',
  name: 'node_023',
  version: '1.1',
  status: 'active',
  priority: 4,
  weight: 0.6959,
  score: 0.0976,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_024',
  name: 'node_024',
  version: '1.6',
  status: 'recovered',
  priority: 8,
  weight: 0.8773,
  score: 0.9578,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_025',
  name: 'node_025',
  version: '4.6',
  status: 'pending',
  priority: 2,
  weight: 0.2248,
  score: 0.7274,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_026',
  name: 'node_026',
  version: '1.0',
  status: 'recovered',
  priority: 6,
  weight: 0.8532,
  score: 0.2915,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_027',
  name: 'node_027',
  version: '2.0',
  status: 'pending',
  priority: 6,
  weight: 0.78,
  score: 0.3969,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_028',
  name: 'node_028',
  version: '2.9',
  status: 'stable',
  priority: 2,
  weight: 0.4741,
  score: 0.0617,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_029',
  name: 'node_029',
  version: '2.0',
  status: 'active',
  priority: 8,
  weight: 0.5888,
  score: 0.4402,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_030',
  name: 'node_030',
  version: '3.3',
  status: 'active',
  priority: 3,
  weight: 0.6374,
  score: 0.2458,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_031',
  name: 'node_031',
  version: '3.3',
  status: 'pending',
  priority: 1,
  weight: 0.4029,
  score: 0.5804,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_032',
  name: 'node_032',
  version: '2.9',
  status: 'completed',
  priority: 10,
  weight: 0.6024,
  score: 0.1325,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_033',
  name: 'node_033',
  version: '3.9',
  status: 'recovered',
  priority: 10,
  weight: 0.273,
  score: 0.6669,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_034',
  name: 'node_034',
  version: '4.3',
  status: 'active',
  priority: 1,
  weight: 0.7475,
  score: 0.5733,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_035',
  name: 'node_035',
  version: '2.7',
  status: 'failed',
  priority: 7,
  weight: 0.7818,
  score: 0.0919,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_036',
  name: 'node_036',
  version: '1.1',
  status: 'recovered',
  priority: 2,
  weight: 0.4573,
  score: 0.0876,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_037',
  name: 'node_037',
  version: '3.6',
  status: 'active',
  priority: 4,
  weight: 0.267,
  score: 0.5992,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_038',
  name: 'node_038',
  version: '3.2',
  status: 'pending',
  priority: 9,
  weight: 0.1035,
  score: 0.0505,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_02_state_handlers_1_039',
  name: 'node_039',
  version: '2.2',
  status: 'recovered',
  priority: 7,
  weight: 0.5884,
  score: 0.4925,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: false
});
