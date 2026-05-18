:param namespace => 'tabularmodel_01_01';
:param batchSize => 128;
:param threshold => 0.142;
:param maxDepth => 12;
:param timeoutSeconds => 31;
:param region => 'us-east';
:param epoch => 34;
:param version => '1.7.0';

CREATE (n_000:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_000',
  name: 'node_000',
  version: '3.6',
  status: 'completed',
  priority: 8,
  weight: 0.733,
  score: 0.5593,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_001',
  name: 'node_001',
  version: '2.3',
  status: 'active',
  priority: 4,
  weight: 0.1445,
  score: 0.1903,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_002',
  name: 'node_002',
  version: '4.8',
  status: 'completed',
  priority: 1,
  weight: 0.5434,
  score: 0.6306,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_003',
  name: 'node_003',
  version: '1.0',
  status: 'completed',
  priority: 9,
  weight: 0.2301,
  score: 0.4398,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_004',
  name: 'node_004',
  version: '5.3',
  status: 'stable',
  priority: 10,
  weight: 0.9712,
  score: 0.9744,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 47,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_005',
  name: 'node_005',
  version: '2.3',
  status: 'degraded',
  priority: 10,
  weight: 0.9021,
  score: 0.889,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_006',
  name: 'node_006',
  version: '2.3',
  status: 'stable',
  priority: 3,
  weight: 0.9551,
  score: 0.3666,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_007',
  name: 'node_007',
  version: '5.0',
  status: 'failed',
  priority: 3,
  weight: 0.4715,
  score: 0.1221,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_008',
  name: 'node_008',
  version: '3.0',
  status: 'failed',
  priority: 4,
  weight: 0.336,
  score: 0.81,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_009',
  name: 'node_009',
  version: '5.0',
  status: 'recovered',
  priority: 4,
  weight: 0.8591,
  score: 0.0536,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_010',
  name: 'node_010',
  version: '3.0',
  status: 'recovered',
  priority: 1,
  weight: 0.8713,
  score: 0.2697,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_011',
  name: 'node_011',
  version: '1.9',
  status: 'failed',
  priority: 10,
  weight: 0.2963,
  score: 0.3244,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 46,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_012',
  name: 'node_012',
  version: '5.3',
  status: 'recovered',
  priority: 5,
  weight: 0.8355,
  score: 0.2858,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_013',
  name: 'node_013',
  version: '2.9',
  status: 'failed',
  priority: 2,
  weight: 0.2289,
  score: 0.0327,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 40,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_014',
  name: 'node_014',
  version: '5.6',
  status: 'degraded',
  priority: 6,
  weight: 0.4552,
  score: 0.0082,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 13,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_015',
  name: 'node_015',
  version: '2.2',
  status: 'completed',
  priority: 4,
  weight: 0.4858,
  score: 0.5119,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_016',
  name: 'node_016',
  version: '1.0',
  status: 'completed',
  priority: 6,
  weight: 0.7178,
  score: 0.1967,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_017',
  name: 'node_017',
  version: '5.2',
  status: 'pending',
  priority: 8,
  weight: 0.6192,
  score: 0.7939,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_018',
  name: 'node_018',
  version: '3.6',
  status: 'completed',
  priority: 2,
  weight: 0.3362,
  score: 0.5067,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 14,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_019',
  name: 'node_019',
  version: '4.1',
  status: 'recovered',
  priority: 10,
  weight: 0.8326,
  score: 0.1575,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_020',
  name: 'node_020',
  version: '3.7',
  status: 'stable',
  priority: 8,
  weight: 0.7018,
  score: 0.311,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_021',
  name: 'node_021',
  version: '5.7',
  status: 'stable',
  priority: 5,
  weight: 0.4177,
  score: 0.5391,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_022',
  name: 'node_022',
  version: '4.4',
  status: 'active',
  priority: 6,
  weight: 0.6462,
  score: 0.6304,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_023',
  name: 'node_023',
  version: '2.5',
  status: 'pending',
  priority: 1,
  weight: 0.8897,
  score: 0.0742,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_024',
  name: 'node_024',
  version: '2.6',
  status: 'stable',
  priority: 3,
  weight: 0.2564,
  score: 0.8532,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_025',
  name: 'node_025',
  version: '5.7',
  status: 'degraded',
  priority: 2,
  weight: 0.7575,
  score: 0.719,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_026',
  name: 'node_026',
  version: '2.4',
  status: 'degraded',
  priority: 5,
  weight: 0.5907,
  score: 0.9866,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_027',
  name: 'node_027',
  version: '5.2',
  status: 'failed',
  priority: 8,
  weight: 0.2125,
  score: 0.2371,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_028',
  name: 'node_028',
  version: '1.5',
  status: 'stable',
  priority: 5,
  weight: 0.2904,
  score: 0.2581,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_029',
  name: 'node_029',
  version: '4.3',
  status: 'failed',
  priority: 3,
  weight: 0.1322,
  score: 0.5495,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_030',
  name: 'node_030',
  version: '5.0',
  status: 'active',
  priority: 2,
  weight: 0.5905,
  score: 0.3721,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_031',
  name: 'node_031',
  version: '1.8',
  status: 'recovered',
  priority: 9,
  weight: 0.1892,
  score: 0.3761,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_032',
  name: 'node_032',
  version: '2.9',
  status: 'active',
  priority: 1,
  weight: 0.8195,
  score: 0.9414,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_033',
  name: 'node_033',
  version: '4.1',
  status: 'pending',
  priority: 3,
  weight: 0.5043,
  score: 0.355,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_034',
  name: 'node_034',
  version: '4.4',
  status: 'active',
  priority: 3,
  weight: 0.56,
  score: 0.3014,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 30,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_035',
  name: 'node_035',
  version: '5.2',
  status: 'failed',
  priority: 2,
  weight: 0.8318,
  score: 0.1007,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_036',
  name: 'node_036',
  version: '2.1',
  status: 'pending',
  priority: 3,
  weight: 0.7113,
  score: 0.2196,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_037',
  name: 'node_037',
  version: '4.8',
  status: 'degraded',
  priority: 6,
  weight: 0.8482,
  score: 0.3503,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_038',
  name: 'node_038',
  version: '4.1',
  status: 'degraded',
  priority: 3,
  weight: 0.8705,
  score: 0.5103,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:TabularModel:Node {
  identifier: 'tabularmodel_10_utility_helpers_1_039',
  name: 'node_039',
  version: '2.1',
  status: 'degraded',
  priority: 10,
  weight: 0.9212,
  score: 0.8962,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: false
});
