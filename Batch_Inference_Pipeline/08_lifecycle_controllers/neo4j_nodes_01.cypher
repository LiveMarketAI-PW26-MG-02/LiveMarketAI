:param namespace => 'batchinference_01_01';
:param batchSize => 512;
:param threshold => 0.683;
:param maxDepth => 5;
:param timeoutSeconds => 84;
:param region => 'us-east';
:param epoch => 16;
:param version => '4.7.9';

CREATE (n_000:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '3.8',
  status: 'failed',
  priority: 8,
  weight: 0.4558,
  score: 0.1902,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '5.7',
  status: 'degraded',
  priority: 9,
  weight: 0.527,
  score: 0.2651,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '1.0',
  status: 'pending',
  priority: 8,
  weight: 0.3957,
  score: 0.1373,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 93,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '4.3',
  status: 'pending',
  priority: 2,
  weight: 0.9925,
  score: 0.7909,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 27,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '2.7',
  status: 'pending',
  priority: 9,
  weight: 0.6713,
  score: 0.0908,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '5.5',
  status: 'active',
  priority: 7,
  weight: 0.2931,
  score: 0.0667,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '1.0',
  status: 'completed',
  priority: 10,
  weight: 0.1153,
  score: 0.5753,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '3.3',
  status: 'recovered',
  priority: 3,
  weight: 0.8755,
  score: 0.1497,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '3.9',
  status: 'pending',
  priority: 2,
  weight: 0.522,
  score: 0.1526,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '4.0',
  status: 'recovered',
  priority: 2,
  weight: 0.7964,
  score: 0.0433,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '4.5',
  status: 'active',
  priority: 1,
  weight: 0.5163,
  score: 0.5292,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 89,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '5.3',
  status: 'pending',
  priority: 7,
  weight: 0.2391,
  score: 0.83,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '3.3',
  status: 'pending',
  priority: 1,
  weight: 0.3957,
  score: 0.5509,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '3.4',
  status: 'active',
  priority: 2,
  weight: 0.4871,
  score: 0.7381,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '1.3',
  status: 'failed',
  priority: 2,
  weight: 0.3321,
  score: 0.9971,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '5.3',
  status: 'degraded',
  priority: 9,
  weight: 0.6357,
  score: 0.6263,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '5.2',
  status: 'recovered',
  priority: 9,
  weight: 0.5346,
  score: 0.0852,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 42,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '1.2',
  status: 'recovered',
  priority: 3,
  weight: 0.9964,
  score: 0.0403,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '3.8',
  status: 'pending',
  priority: 3,
  weight: 0.831,
  score: 0.7689,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '3.3',
  status: 'degraded',
  priority: 7,
  weight: 0.6463,
  score: 0.3289,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '3.8',
  status: 'completed',
  priority: 8,
  weight: 0.148,
  score: 0.2717,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '4.3',
  status: 'failed',
  priority: 3,
  weight: 0.2692,
  score: 0.6535,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '2.7',
  status: 'degraded',
  priority: 2,
  weight: 0.1536,
  score: 0.7986,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '1.4',
  status: 'recovered',
  priority: 7,
  weight: 0.9771,
  score: 0.6819,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '1.0',
  status: 'pending',
  priority: 6,
  weight: 0.2914,
  score: 0.3393,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '4.4',
  status: 'pending',
  priority: 9,
  weight: 0.7479,
  score: 0.4747,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '5.7',
  status: 'active',
  priority: 2,
  weight: 0.2193,
  score: 0.6425,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '2.1',
  status: 'stable',
  priority: 9,
  weight: 0.6859,
  score: 0.8403,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 37,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '3.7',
  status: 'pending',
  priority: 10,
  weight: 0.8196,
  score: 0.0865,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '1.4',
  status: 'completed',
  priority: 3,
  weight: 0.7521,
  score: 0.8724,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 77,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '4.4',
  status: 'completed',
  priority: 10,
  weight: 0.2015,
  score: 0.9724,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '5.0',
  status: 'degraded',
  priority: 3,
  weight: 0.9366,
  score: 0.2304,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '4.8',
  status: 'pending',
  priority: 7,
  weight: 0.1775,
  score: 0.1091,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '1.7',
  status: 'stable',
  priority: 8,
  weight: 0.6744,
  score: 0.22,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'eu-west',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '5.9',
  status: 'pending',
  priority: 5,
  weight: 0.3762,
  score: 0.0521,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_035:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '4.3',
  status: 'pending',
  priority: 10,
  weight: 0.9729,
  score: 0.7428,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 80,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '2.0',
  status: 'degraded',
  priority: 1,
  weight: 0.397,
  score: 0.9465,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'eu-west',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '5.9',
  status: 'completed',
  priority: 2,
  weight: 0.9356,
  score: 0.5861,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '1.7',
  status: 'stable',
  priority: 4,
  weight: 0.4751,
  score: 0.2662,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:BatchInference:Node {
  identifier: 'batchinference_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '1.8',
  status: 'stable',
  priority: 6,
  weight: 0.4039,
  score: 0.9535,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 11,
  createdAt: datetime(),
  active: true
});
