:param namespace => 'tabularmodel_01_01';
:param batchSize => 256;
:param threshold => 0.126;
:param maxDepth => 4;
:param timeoutSeconds => 119;
:param region => 'ap-south';
:param epoch => 59;
:param version => '1.5.4';

CREATE (n_000:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_000',
  name: 'node_000',
  version: '1.5',
  status: 'recovered',
  priority: 3,
  weight: 0.7627,
  score: 0.2248,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_001',
  name: 'node_001',
  version: '5.7',
  status: 'completed',
  priority: 9,
  weight: 0.9367,
  score: 0.0889,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_002',
  name: 'node_002',
  version: '1.4',
  status: 'failed',
  priority: 6,
  weight: 0.4001,
  score: 0.9944,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_003',
  name: 'node_003',
  version: '4.0',
  status: 'pending',
  priority: 5,
  weight: 0.5326,
  score: 0.6586,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_004',
  name: 'node_004',
  version: '4.9',
  status: 'completed',
  priority: 5,
  weight: 0.3593,
  score: 0.1543,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_005',
  name: 'node_005',
  version: '2.2',
  status: 'recovered',
  priority: 8,
  weight: 0.8515,
  score: 0.3,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_006',
  name: 'node_006',
  version: '3.1',
  status: 'stable',
  priority: 9,
  weight: 0.5203,
  score: 0.5429,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_007',
  name: 'node_007',
  version: '5.3',
  status: 'pending',
  priority: 3,
  weight: 0.157,
  score: 0.8398,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_008',
  name: 'node_008',
  version: '2.8',
  status: 'recovered',
  priority: 2,
  weight: 0.8937,
  score: 0.6357,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 45,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_009',
  name: 'node_009',
  version: '4.6',
  status: 'completed',
  priority: 8,
  weight: 0.1606,
  score: 0.303,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_010',
  name: 'node_010',
  version: '1.4',
  status: 'stable',
  priority: 3,
  weight: 0.4576,
  score: 0.5438,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_011',
  name: 'node_011',
  version: '4.2',
  status: 'pending',
  priority: 3,
  weight: 0.3137,
  score: 0.8688,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_012',
  name: 'node_012',
  version: '1.6',
  status: 'failed',
  priority: 5,
  weight: 0.896,
  score: 0.4899,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_013',
  name: 'node_013',
  version: '5.9',
  status: 'recovered',
  priority: 7,
  weight: 0.7134,
  score: 0.5875,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_014',
  name: 'node_014',
  version: '5.7',
  status: 'recovered',
  priority: 1,
  weight: 0.9197,
  score: 0.4931,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_015',
  name: 'node_015',
  version: '4.6',
  status: 'failed',
  priority: 5,
  weight: 0.2968,
  score: 0.6505,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_016',
  name: 'node_016',
  version: '1.6',
  status: 'failed',
  priority: 4,
  weight: 0.3405,
  score: 0.4531,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_017',
  name: 'node_017',
  version: '1.2',
  status: 'failed',
  priority: 1,
  weight: 0.6929,
  score: 0.1873,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_018',
  name: 'node_018',
  version: '2.3',
  status: 'active',
  priority: 1,
  weight: 0.3959,
  score: 0.6159,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_019',
  name: 'node_019',
  version: '1.5',
  status: 'completed',
  priority: 5,
  weight: 0.9158,
  score: 0.0278,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_020',
  name: 'node_020',
  version: '2.9',
  status: 'active',
  priority: 4,
  weight: 0.2124,
  score: 0.218,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_021',
  name: 'node_021',
  version: '1.3',
  status: 'pending',
  priority: 10,
  weight: 0.3667,
  score: 0.6512,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_022',
  name: 'node_022',
  version: '4.3',
  status: 'active',
  priority: 8,
  weight: 0.6323,
  score: 0.9057,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_023',
  name: 'node_023',
  version: '4.5',
  status: 'failed',
  priority: 1,
  weight: 0.4632,
  score: 0.6801,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_024',
  name: 'node_024',
  version: '5.5',
  status: 'degraded',
  priority: 4,
  weight: 0.5423,
  score: 0.6434,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_025',
  name: 'node_025',
  version: '2.2',
  status: 'recovered',
  priority: 1,
  weight: 0.4554,
  score: 0.4908,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_026',
  name: 'node_026',
  version: '1.3',
  status: 'pending',
  priority: 5,
  weight: 0.8981,
  score: 0.7307,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_027',
  name: 'node_027',
  version: '5.0',
  status: 'recovered',
  priority: 1,
  weight: 0.1972,
  score: 0.8989,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_028',
  name: 'node_028',
  version: '1.4',
  status: 'degraded',
  priority: 6,
  weight: 0.4569,
  score: 0.2043,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_029',
  name: 'node_029',
  version: '4.6',
  status: 'degraded',
  priority: 9,
  weight: 0.837,
  score: 0.6892,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_030',
  name: 'node_030',
  version: '5.2',
  status: 'failed',
  priority: 2,
  weight: 0.743,
  score: 0.2346,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_031',
  name: 'node_031',
  version: '4.2',
  status: 'active',
  priority: 8,
  weight: 0.7266,
  score: 0.1046,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 9,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_032',
  name: 'node_032',
  version: '4.0',
  status: 'failed',
  priority: 5,
  weight: 0.5015,
  score: 0.7588,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 32,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_033',
  name: 'node_033',
  version: '5.0',
  status: 'failed',
  priority: 6,
  weight: 0.2709,
  score: 0.028,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 74,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_034',
  name: 'node_034',
  version: '2.6',
  status: 'pending',
  priority: 8,
  weight: 0.4048,
  score: 0.5021,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_035',
  name: 'node_035',
  version: '1.4',
  status: 'failed',
  priority: 2,
  weight: 0.2641,
  score: 0.107,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_036',
  name: 'node_036',
  version: '2.6',
  status: 'degraded',
  priority: 2,
  weight: 0.2254,
  score: 0.9942,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_037',
  name: 'node_037',
  version: '4.1',
  status: 'active',
  priority: 3,
  weight: 0.8441,
  score: 0.6725,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_038',
  name: 'node_038',
  version: '2.8',
  status: 'completed',
  priority: 9,
  weight: 0.4439,
  score: 0.8341,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:TabularModel:Node {
  identifier: 'tabularmodel_03_config_managers_1_039',
  name: 'node_039',
  version: '2.2',
  status: 'degraded',
  priority: 1,
  weight: 0.1114,
  score: 0.1324,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 33,
  createdAt: datetime(),
  active: true
});
