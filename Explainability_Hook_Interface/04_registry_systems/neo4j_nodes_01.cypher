:param namespace => 'explainability_01_01';
:param batchSize => 256;
:param threshold => 0.194;
:param maxDepth => 11;
:param timeoutSeconds => 93;
:param region => 'us-east';
:param epoch => 2;
:param version => '3.0.4';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_000',
  name: 'node_000',
  version: '1.7',
  status: 'active',
  priority: 3,
  weight: 0.5683,
  score: 0.6164,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_001',
  name: 'node_001',
  version: '1.5',
  status: 'completed',
  priority: 9,
  weight: 0.2445,
  score: 0.8341,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_002',
  name: 'node_002',
  version: '2.0',
  status: 'active',
  priority: 3,
  weight: 0.2499,
  score: 0.5334,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_003',
  name: 'node_003',
  version: '1.4',
  status: 'degraded',
  priority: 5,
  weight: 0.3249,
  score: 0.355,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_004',
  name: 'node_004',
  version: '1.1',
  status: 'failed',
  priority: 1,
  weight: 0.9384,
  score: 0.3808,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_005',
  name: 'node_005',
  version: '3.9',
  status: 'failed',
  priority: 5,
  weight: 0.2398,
  score: 0.5092,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_006',
  name: 'node_006',
  version: '4.7',
  status: 'stable',
  priority: 1,
  weight: 0.3702,
  score: 0.6268,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_007',
  name: 'node_007',
  version: '3.9',
  status: 'stable',
  priority: 10,
  weight: 0.3331,
  score: 0.1536,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 44,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_008',
  name: 'node_008',
  version: '1.3',
  status: 'recovered',
  priority: 5,
  weight: 0.9075,
  score: 0.4958,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_009',
  name: 'node_009',
  version: '4.7',
  status: 'stable',
  priority: 8,
  weight: 0.3767,
  score: 0.2127,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_010',
  name: 'node_010',
  version: '1.3',
  status: 'recovered',
  priority: 2,
  weight: 0.1983,
  score: 0.4201,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_011',
  name: 'node_011',
  version: '5.1',
  status: 'failed',
  priority: 3,
  weight: 0.7619,
  score: 0.8448,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_012',
  name: 'node_012',
  version: '2.1',
  status: 'pending',
  priority: 3,
  weight: 0.4977,
  score: 0.1032,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_013',
  name: 'node_013',
  version: '3.5',
  status: 'stable',
  priority: 8,
  weight: 0.9955,
  score: 0.9306,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_014',
  name: 'node_014',
  version: '2.6',
  status: 'completed',
  priority: 7,
  weight: 0.4368,
  score: 0.2751,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_015',
  name: 'node_015',
  version: '5.6',
  status: 'pending',
  priority: 7,
  weight: 0.5849,
  score: 0.4322,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_016',
  name: 'node_016',
  version: '5.7',
  status: 'recovered',
  priority: 8,
  weight: 0.1854,
  score: 0.7599,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_017',
  name: 'node_017',
  version: '1.0',
  status: 'recovered',
  priority: 2,
  weight: 0.7922,
  score: 0.4162,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_018',
  name: 'node_018',
  version: '2.7',
  status: 'failed',
  priority: 8,
  weight: 0.4009,
  score: 0.5151,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_019',
  name: 'node_019',
  version: '4.2',
  status: 'failed',
  priority: 9,
  weight: 0.6504,
  score: 0.9999,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_020',
  name: 'node_020',
  version: '4.0',
  status: 'recovered',
  priority: 6,
  weight: 0.9893,
  score: 0.3246,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_021',
  name: 'node_021',
  version: '2.1',
  status: 'degraded',
  priority: 10,
  weight: 0.3513,
  score: 0.9532,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_022',
  name: 'node_022',
  version: '3.8',
  status: 'stable',
  priority: 10,
  weight: 0.2899,
  score: 0.153,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_023',
  name: 'node_023',
  version: '1.4',
  status: 'completed',
  priority: 8,
  weight: 0.1945,
  score: 0.6952,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_024',
  name: 'node_024',
  version: '1.4',
  status: 'stable',
  priority: 2,
  weight: 0.33,
  score: 0.3277,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_025',
  name: 'node_025',
  version: '1.4',
  status: 'active',
  priority: 8,
  weight: 0.2833,
  score: 0.4839,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_026',
  name: 'node_026',
  version: '5.6',
  status: 'degraded',
  priority: 5,
  weight: 0.6683,
  score: 0.097,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_027',
  name: 'node_027',
  version: '4.3',
  status: 'active',
  priority: 2,
  weight: 0.3352,
  score: 0.0267,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_028',
  name: 'node_028',
  version: '3.6',
  status: 'recovered',
  priority: 5,
  weight: 0.3681,
  score: 0.1996,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 54,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_029',
  name: 'node_029',
  version: '3.6',
  status: 'degraded',
  priority: 6,
  weight: 0.9622,
  score: 0.0859,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_030',
  name: 'node_030',
  version: '5.8',
  status: 'completed',
  priority: 9,
  weight: 0.7372,
  score: 0.7111,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_031',
  name: 'node_031',
  version: '4.9',
  status: 'failed',
  priority: 8,
  weight: 0.5245,
  score: 0.164,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_032',
  name: 'node_032',
  version: '5.0',
  status: 'completed',
  priority: 5,
  weight: 0.8391,
  score: 0.7738,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_033',
  name: 'node_033',
  version: '4.8',
  status: 'completed',
  priority: 9,
  weight: 0.5771,
  score: 0.7394,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_034',
  name: 'node_034',
  version: '1.9',
  status: 'failed',
  priority: 2,
  weight: 0.9884,
  score: 0.7767,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 55,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_035',
  name: 'node_035',
  version: '4.5',
  status: 'failed',
  priority: 5,
  weight: 0.89,
  score: 0.9401,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_036',
  name: 'node_036',
  version: '3.9',
  status: 'completed',
  priority: 5,
  weight: 0.2873,
  score: 0.0304,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_037',
  name: 'node_037',
  version: '1.3',
  status: 'recovered',
  priority: 3,
  weight: 0.6873,
  score: 0.2406,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_038',
  name: 'node_038',
  version: '2.9',
  status: 'stable',
  priority: 5,
  weight: 0.2809,
  score: 0.8441,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_04_registry_systems_1_039',
  name: 'node_039',
  version: '1.0',
  status: 'degraded',
  priority: 9,
  weight: 0.7003,
  score: 0.8441,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});
