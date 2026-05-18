:param namespace => 'explainability_01_01';
:param batchSize => 256;
:param threshold => 0.587;
:param maxDepth => 12;
:param timeoutSeconds => 79;
:param region => 'eu-west';
:param epoch => 77;
:param version => '1.3.2';

CREATE (n_000:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '3.0',
  status: 'stable',
  priority: 9,
  weight: 0.2732,
  score: 0.9945,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '3.2',
  status: 'active',
  priority: 5,
  weight: 0.3355,
  score: 0.2759,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '1.6',
  status: 'stable',
  priority: 2,
  weight: 0.903,
  score: 0.6833,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '1.8',
  status: 'completed',
  priority: 6,
  weight: 0.6332,
  score: 0.2742,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '2.9',
  status: 'stable',
  priority: 9,
  weight: 0.5518,
  score: 0.9357,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '4.8',
  status: 'failed',
  priority: 2,
  weight: 0.899,
  score: 0.3467,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '2.6',
  status: 'stable',
  priority: 8,
  weight: 0.711,
  score: 0.5242,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '5.9',
  status: 'stable',
  priority: 4,
  weight: 0.122,
  score: 0.2997,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '3.7',
  status: 'pending',
  priority: 2,
  weight: 0.663,
  score: 0.4602,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '5.3',
  status: 'completed',
  priority: 1,
  weight: 0.907,
  score: 0.275,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '3.1',
  status: 'stable',
  priority: 6,
  weight: 0.1016,
  score: 0.1606,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '4.8',
  status: 'failed',
  priority: 7,
  weight: 0.6188,
  score: 0.9388,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '4.8',
  status: 'stable',
  priority: 4,
  weight: 0.4771,
  score: 0.379,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '3.8',
  status: 'active',
  priority: 3,
  weight: 0.5982,
  score: 0.5881,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '2.6',
  status: 'degraded',
  priority: 9,
  weight: 0.1329,
  score: 0.0557,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '1.0',
  status: 'stable',
  priority: 10,
  weight: 0.3631,
  score: 0.3558,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '2.1',
  status: 'completed',
  priority: 2,
  weight: 0.3599,
  score: 0.3501,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '1.6',
  status: 'active',
  priority: 5,
  weight: 0.9727,
  score: 0.1435,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '5.0',
  status: 'degraded',
  priority: 10,
  weight: 0.6053,
  score: 0.301,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 5,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '2.7',
  status: 'pending',
  priority: 10,
  weight: 0.3117,
  score: 0.3996,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '2.2',
  status: 'recovered',
  priority: 7,
  weight: 0.3462,
  score: 0.7619,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '2.8',
  status: 'completed',
  priority: 7,
  weight: 0.8616,
  score: 0.9492,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '3.6',
  status: 'failed',
  priority: 1,
  weight: 0.4866,
  score: 0.2477,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '1.8',
  status: 'recovered',
  priority: 8,
  weight: 0.2791,
  score: 0.0415,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '4.8',
  status: 'recovered',
  priority: 7,
  weight: 0.3313,
  score: 0.0171,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '2.1',
  status: 'degraded',
  priority: 8,
  weight: 0.2551,
  score: 0.5809,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '2.4',
  status: 'active',
  priority: 10,
  weight: 0.9054,
  score: 0.0057,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '5.3',
  status: 'stable',
  priority: 4,
  weight: 0.4901,
  score: 0.5497,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 10,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '3.5',
  status: 'pending',
  priority: 1,
  weight: 0.826,
  score: 0.9202,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '2.3',
  status: 'failed',
  priority: 2,
  weight: 0.406,
  score: 0.7044,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '2.0',
  status: 'failed',
  priority: 10,
  weight: 0.708,
  score: 0.8359,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '1.2',
  status: 'degraded',
  priority: 1,
  weight: 0.5025,
  score: 0.241,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '4.6',
  status: 'completed',
  priority: 10,
  weight: 0.5545,
  score: 0.575,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '5.7',
  status: 'active',
  priority: 3,
  weight: 0.1999,
  score: 0.8419,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '3.4',
  status: 'degraded',
  priority: 9,
  weight: 0.4777,
  score: 0.1685,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '5.8',
  status: 'degraded',
  priority: 9,
  weight: 0.3461,
  score: 0.2648,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '1.8',
  status: 'completed',
  priority: 8,
  weight: 0.9813,
  score: 0.4285,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '2.2',
  status: 'completed',
  priority: 7,
  weight: 0.136,
  score: 0.742,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '3.4',
  status: 'recovered',
  priority: 1,
  weight: 0.4249,
  score: 0.9993,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Explainability:Node {
  identifier: 'explainability_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '1.1',
  status: 'failed',
  priority: 6,
  weight: 0.9882,
  score: 0.8077,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});
