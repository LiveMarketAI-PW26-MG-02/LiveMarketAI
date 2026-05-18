:param namespace => 'uncertainty_01_01';
:param batchSize => 64;
:param threshold => 0.549;
:param maxDepth => 9;
:param timeoutSeconds => 46;
:param region => 'ap-south';
:param epoch => 48;
:param version => '5.8.4';

CREATE (n_000:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_000',
  name: 'node_000',
  version: '2.2',
  status: 'stable',
  priority: 9,
  weight: 0.4005,
  score: 0.589,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_001',
  name: 'node_001',
  version: '2.2',
  status: 'completed',
  priority: 1,
  weight: 0.5638,
  score: 0.0009,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_002',
  name: 'node_002',
  version: '3.7',
  status: 'completed',
  priority: 9,
  weight: 0.5608,
  score: 0.0865,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 5,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_003',
  name: 'node_003',
  version: '2.6',
  status: 'completed',
  priority: 1,
  weight: 0.2933,
  score: 0.6119,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_004',
  name: 'node_004',
  version: '5.1',
  status: 'active',
  priority: 5,
  weight: 0.5116,
  score: 0.0341,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_005',
  name: 'node_005',
  version: '4.1',
  status: 'recovered',
  priority: 10,
  weight: 0.536,
  score: 0.487,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_006',
  name: 'node_006',
  version: '5.3',
  status: 'completed',
  priority: 2,
  weight: 0.5424,
  score: 0.5033,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_007',
  name: 'node_007',
  version: '4.3',
  status: 'pending',
  priority: 9,
  weight: 0.4314,
  score: 0.6113,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_008',
  name: 'node_008',
  version: '2.0',
  status: 'degraded',
  priority: 7,
  weight: 0.488,
  score: 0.1483,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_009',
  name: 'node_009',
  version: '3.4',
  status: 'completed',
  priority: 8,
  weight: 0.8487,
  score: 0.436,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_010',
  name: 'node_010',
  version: '5.7',
  status: 'failed',
  priority: 10,
  weight: 0.304,
  score: 0.9926,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_011',
  name: 'node_011',
  version: '1.1',
  status: 'pending',
  priority: 6,
  weight: 0.53,
  score: 0.2038,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_012',
  name: 'node_012',
  version: '1.5',
  status: 'active',
  priority: 2,
  weight: 0.809,
  score: 0.1279,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_013',
  name: 'node_013',
  version: '1.5',
  status: 'failed',
  priority: 2,
  weight: 0.7802,
  score: 0.9992,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_014',
  name: 'node_014',
  version: '4.6',
  status: 'stable',
  priority: 2,
  weight: 0.717,
  score: 0.8317,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_015',
  name: 'node_015',
  version: '1.3',
  status: 'stable',
  priority: 10,
  weight: 0.9889,
  score: 0.4606,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 86,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_016',
  name: 'node_016',
  version: '5.1',
  status: 'active',
  priority: 6,
  weight: 0.8297,
  score: 0.2519,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_017',
  name: 'node_017',
  version: '1.2',
  status: 'failed',
  priority: 1,
  weight: 0.2946,
  score: 0.6235,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_018',
  name: 'node_018',
  version: '1.7',
  status: 'active',
  priority: 7,
  weight: 0.9785,
  score: 0.1158,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_019',
  name: 'node_019',
  version: '5.5',
  status: 'degraded',
  priority: 3,
  weight: 0.7687,
  score: 0.7784,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_020',
  name: 'node_020',
  version: '4.7',
  status: 'stable',
  priority: 3,
  weight: 0.8079,
  score: 0.0017,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_021',
  name: 'node_021',
  version: '1.5',
  status: 'active',
  priority: 2,
  weight: 0.471,
  score: 0.4106,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_022',
  name: 'node_022',
  version: '1.6',
  status: 'active',
  priority: 6,
  weight: 0.7261,
  score: 0.7149,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_023',
  name: 'node_023',
  version: '4.0',
  status: 'degraded',
  priority: 9,
  weight: 0.2813,
  score: 0.0133,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_024',
  name: 'node_024',
  version: '1.2',
  status: 'degraded',
  priority: 3,
  weight: 0.3257,
  score: 0.4882,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_025',
  name: 'node_025',
  version: '3.4',
  status: 'degraded',
  priority: 6,
  weight: 0.7806,
  score: 0.7955,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_026',
  name: 'node_026',
  version: '2.5',
  status: 'completed',
  priority: 3,
  weight: 0.1189,
  score: 0.8897,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_027',
  name: 'node_027',
  version: '1.9',
  status: 'failed',
  priority: 9,
  weight: 0.6784,
  score: 0.0584,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_028',
  name: 'node_028',
  version: '2.9',
  status: 'completed',
  priority: 5,
  weight: 0.196,
  score: 0.0856,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 18,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_029',
  name: 'node_029',
  version: '4.4',
  status: 'completed',
  priority: 7,
  weight: 0.5995,
  score: 0.9781,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_030',
  name: 'node_030',
  version: '2.9',
  status: 'stable',
  priority: 6,
  weight: 0.8846,
  score: 0.5584,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_031',
  name: 'node_031',
  version: '4.7',
  status: 'pending',
  priority: 9,
  weight: 0.8406,
  score: 0.4603,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_032',
  name: 'node_032',
  version: '2.3',
  status: 'active',
  priority: 5,
  weight: 0.7845,
  score: 0.7763,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_033',
  name: 'node_033',
  version: '1.5',
  status: 'degraded',
  priority: 7,
  weight: 0.563,
  score: 0.3767,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_034',
  name: 'node_034',
  version: '3.2',
  status: 'stable',
  priority: 2,
  weight: 0.8265,
  score: 0.084,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_035',
  name: 'node_035',
  version: '3.0',
  status: 'recovered',
  priority: 3,
  weight: 0.7768,
  score: 0.9423,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_036',
  name: 'node_036',
  version: '4.5',
  status: 'completed',
  priority: 9,
  weight: 0.6234,
  score: 0.3615,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_037',
  name: 'node_037',
  version: '2.2',
  status: 'active',
  priority: 6,
  weight: 0.8006,
  score: 0.0346,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_038',
  name: 'node_038',
  version: '2.3',
  status: 'recovered',
  priority: 7,
  weight: 0.642,
  score: 0.1017,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:Uncertainty:Node {
  identifier: 'uncertainty_04_registry_systems_1_039',
  name: 'node_039',
  version: '3.6',
  status: 'recovered',
  priority: 8,
  weight: 0.2966,
  score: 0.1341,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});
