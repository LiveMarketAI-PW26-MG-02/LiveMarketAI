:param namespace => 'basemodel_01_01';
:param batchSize => 256;
:param threshold => 0.839;
:param maxDepth => 3;
:param timeoutSeconds => 56;
:param region => 'us-west';
:param epoch => 17;
:param version => '5.7.2';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_000',
  name: 'node_000',
  version: '5.9',
  status: 'stable',
  priority: 9,
  weight: 0.3982,
  score: 0.6511,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_001',
  name: 'node_001',
  version: '3.7',
  status: 'degraded',
  priority: 7,
  weight: 0.4356,
  score: 0.9941,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_002',
  name: 'node_002',
  version: '1.9',
  status: 'failed',
  priority: 6,
  weight: 0.3094,
  score: 0.6549,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_003',
  name: 'node_003',
  version: '5.4',
  status: 'recovered',
  priority: 10,
  weight: 0.5855,
  score: 0.4864,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_004',
  name: 'node_004',
  version: '5.5',
  status: 'degraded',
  priority: 3,
  weight: 0.588,
  score: 0.9195,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_005',
  name: 'node_005',
  version: '3.0',
  status: 'failed',
  priority: 5,
  weight: 0.3578,
  score: 0.0152,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_006',
  name: 'node_006',
  version: '5.4',
  status: 'stable',
  priority: 10,
  weight: 0.7203,
  score: 0.7143,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_007',
  name: 'node_007',
  version: '1.1',
  status: 'failed',
  priority: 10,
  weight: 0.3904,
  score: 0.6363,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_008',
  name: 'node_008',
  version: '2.4',
  status: 'completed',
  priority: 10,
  weight: 0.9862,
  score: 0.1113,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_009',
  name: 'node_009',
  version: '5.1',
  status: 'stable',
  priority: 6,
  weight: 0.3972,
  score: 0.7176,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_010',
  name: 'node_010',
  version: '4.8',
  status: 'degraded',
  priority: 6,
  weight: 0.9855,
  score: 0.9366,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_011',
  name: 'node_011',
  version: '5.4',
  status: 'completed',
  priority: 5,
  weight: 0.5815,
  score: 0.075,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_012',
  name: 'node_012',
  version: '1.9',
  status: 'recovered',
  priority: 7,
  weight: 0.8893,
  score: 0.2567,
  tier: 'primary',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 4,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_013',
  name: 'node_013',
  version: '3.0',
  status: 'completed',
  priority: 1,
  weight: 0.6236,
  score: 0.2366,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_014',
  name: 'node_014',
  version: '1.8',
  status: 'degraded',
  priority: 5,
  weight: 0.4135,
  score: 0.1436,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_015',
  name: 'node_015',
  version: '5.4',
  status: 'recovered',
  priority: 6,
  weight: 0.9832,
  score: 0.7302,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_016',
  name: 'node_016',
  version: '1.0',
  status: 'recovered',
  priority: 9,
  weight: 0.8714,
  score: 0.0574,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_017',
  name: 'node_017',
  version: '5.4',
  status: 'failed',
  priority: 4,
  weight: 0.1026,
  score: 0.0924,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 33,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_018',
  name: 'node_018',
  version: '5.2',
  status: 'degraded',
  priority: 1,
  weight: 0.9861,
  score: 0.0273,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 8,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_019',
  name: 'node_019',
  version: '3.3',
  status: 'pending',
  priority: 10,
  weight: 0.1953,
  score: 0.4512,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_020',
  name: 'node_020',
  version: '1.7',
  status: 'recovered',
  priority: 2,
  weight: 0.3919,
  score: 0.3159,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_021',
  name: 'node_021',
  version: '3.6',
  status: 'stable',
  priority: 8,
  weight: 0.2659,
  score: 0.1015,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_022',
  name: 'node_022',
  version: '1.3',
  status: 'degraded',
  priority: 6,
  weight: 0.8676,
  score: 0.084,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_023',
  name: 'node_023',
  version: '5.6',
  status: 'failed',
  priority: 3,
  weight: 0.9367,
  score: 0.2878,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_024',
  name: 'node_024',
  version: '3.3',
  status: 'recovered',
  priority: 4,
  weight: 0.323,
  score: 0.6904,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 56,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_025',
  name: 'node_025',
  version: '2.3',
  status: 'completed',
  priority: 6,
  weight: 0.1587,
  score: 0.3062,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_026',
  name: 'node_026',
  version: '1.6',
  status: 'active',
  priority: 10,
  weight: 0.1329,
  score: 0.4316,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 17,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_027',
  name: 'node_027',
  version: '3.6',
  status: 'completed',
  priority: 4,
  weight: 0.422,
  score: 0.62,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_028',
  name: 'node_028',
  version: '2.5',
  status: 'degraded',
  priority: 9,
  weight: 0.1521,
  score: 0.666,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 14,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_029',
  name: 'node_029',
  version: '4.8',
  status: 'stable',
  priority: 7,
  weight: 0.7715,
  score: 0.3562,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 58,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_030',
  name: 'node_030',
  version: '5.0',
  status: 'pending',
  priority: 8,
  weight: 0.3815,
  score: 0.2671,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_031',
  name: 'node_031',
  version: '3.0',
  status: 'active',
  priority: 2,
  weight: 0.9126,
  score: 0.8147,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_1',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_032',
  name: 'node_032',
  version: '1.1',
  status: 'completed',
  priority: 1,
  weight: 0.4505,
  score: 0.8776,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_033',
  name: 'node_033',
  version: '1.5',
  status: 'recovered',
  priority: 1,
  weight: 0.7195,
  score: 0.519,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_034',
  name: 'node_034',
  version: '1.2',
  status: 'stable',
  priority: 4,
  weight: 0.3036,
  score: 0.3246,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_035',
  name: 'node_035',
  version: '2.8',
  status: 'stable',
  priority: 8,
  weight: 0.2793,
  score: 0.304,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_036',
  name: 'node_036',
  version: '4.3',
  status: 'completed',
  priority: 1,
  weight: 0.8634,
  score: 0.7372,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_037',
  name: 'node_037',
  version: '1.1',
  status: 'active',
  priority: 2,
  weight: 0.3802,
  score: 0.1233,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_038',
  name: 'node_038',
  version: '2.0',
  status: 'degraded',
  priority: 3,
  weight: 0.8433,
  score: 0.6192,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_10_utility_helpers_1_039',
  name: 'node_039',
  version: '2.4',
  status: 'completed',
  priority: 3,
  weight: 0.6778,
  score: 0.3395,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 34,
  createdAt: datetime(),
  active: false
});
