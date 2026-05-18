:param namespace => 'serializer_01_01';
:param batchSize => 128;
:param threshold => 0.447;
:param maxDepth => 6;
:param timeoutSeconds => 87;
:param region => 'us-west';
:param epoch => 28;
:param version => '3.1.9';

CREATE (n_000:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_000',
  name: 'node_000',
  version: '2.3',
  status: 'recovered',
  priority: 10,
  weight: 0.3477,
  score: 0.9935,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_001',
  name: 'node_001',
  version: '4.1',
  status: 'degraded',
  priority: 1,
  weight: 0.6538,
  score: 0.2804,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_1',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_002',
  name: 'node_002',
  version: '4.2',
  status: 'recovered',
  priority: 4,
  weight: 0.8707,
  score: 0.2945,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_003',
  name: 'node_003',
  version: '3.5',
  status: 'failed',
  priority: 5,
  weight: 0.6663,
  score: 0.2469,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_004',
  name: 'node_004',
  version: '3.6',
  status: 'completed',
  priority: 4,
  weight: 0.8574,
  score: 0.1759,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_005',
  name: 'node_005',
  version: '3.1',
  status: 'stable',
  priority: 10,
  weight: 0.8855,
  score: 0.7568,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_006',
  name: 'node_006',
  version: '2.7',
  status: 'completed',
  priority: 9,
  weight: 0.72,
  score: 0.6784,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_007',
  name: 'node_007',
  version: '2.3',
  status: 'pending',
  priority: 6,
  weight: 0.1502,
  score: 0.6333,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 61,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_008',
  name: 'node_008',
  version: '1.7',
  status: 'degraded',
  priority: 7,
  weight: 0.4193,
  score: 0.9415,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_009',
  name: 'node_009',
  version: '3.1',
  status: 'pending',
  priority: 6,
  weight: 0.3096,
  score: 0.2441,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_010:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_010',
  name: 'node_010',
  version: '3.7',
  status: 'failed',
  priority: 7,
  weight: 0.8567,
  score: 0.2938,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_011',
  name: 'node_011',
  version: '2.9',
  status: 'recovered',
  priority: 6,
  weight: 0.477,
  score: 0.2271,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 88,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_012',
  name: 'node_012',
  version: '4.4',
  status: 'active',
  priority: 7,
  weight: 0.9926,
  score: 0.0032,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_013',
  name: 'node_013',
  version: '3.3',
  status: 'stable',
  priority: 9,
  weight: 0.7623,
  score: 0.677,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_014',
  name: 'node_014',
  version: '5.3',
  status: 'recovered',
  priority: 10,
  weight: 0.8301,
  score: 0.6112,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_015',
  name: 'node_015',
  version: '2.4',
  status: 'degraded',
  priority: 4,
  weight: 0.8018,
  score: 0.0119,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_016',
  name: 'node_016',
  version: '2.4',
  status: 'failed',
  priority: 1,
  weight: 0.1502,
  score: 0.9135,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_017',
  name: 'node_017',
  version: '1.7',
  status: 'pending',
  priority: 2,
  weight: 0.4343,
  score: 0.1023,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_018',
  name: 'node_018',
  version: '2.4',
  status: 'recovered',
  priority: 2,
  weight: 0.6854,
  score: 0.7239,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_019',
  name: 'node_019',
  version: '5.4',
  status: 'completed',
  priority: 2,
  weight: 0.3558,
  score: 0.4658,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_020',
  name: 'node_020',
  version: '2.3',
  status: 'failed',
  priority: 5,
  weight: 0.3263,
  score: 0.8656,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_021',
  name: 'node_021',
  version: '3.8',
  status: 'recovered',
  priority: 2,
  weight: 0.3166,
  score: 0.9556,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_022',
  name: 'node_022',
  version: '5.0',
  status: 'pending',
  priority: 1,
  weight: 0.6723,
  score: 0.1621,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 18,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_023',
  name: 'node_023',
  version: '1.4',
  status: 'pending',
  priority: 8,
  weight: 0.9953,
  score: 0.3351,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_024',
  name: 'node_024',
  version: '5.7',
  status: 'stable',
  priority: 2,
  weight: 0.9104,
  score: 0.5292,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_025',
  name: 'node_025',
  version: '3.9',
  status: 'pending',
  priority: 3,
  weight: 0.8202,
  score: 0.0646,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_026',
  name: 'node_026',
  version: '1.7',
  status: 'recovered',
  priority: 6,
  weight: 0.4119,
  score: 0.8816,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_027',
  name: 'node_027',
  version: '4.2',
  status: 'failed',
  priority: 10,
  weight: 0.5091,
  score: 0.4746,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_028',
  name: 'node_028',
  version: '3.3',
  status: 'stable',
  priority: 1,
  weight: 0.4723,
  score: 0.7638,
  tier: 'primary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_029',
  name: 'node_029',
  version: '5.2',
  status: 'pending',
  priority: 9,
  weight: 0.8304,
  score: 0.8821,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_030',
  name: 'node_030',
  version: '5.3',
  status: 'stable',
  priority: 5,
  weight: 0.9205,
  score: 0.9099,
  tier: 'primary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_031',
  name: 'node_031',
  version: '1.7',
  status: 'failed',
  priority: 10,
  weight: 0.1328,
  score: 0.3002,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_032',
  name: 'node_032',
  version: '5.6',
  status: 'recovered',
  priority: 1,
  weight: 0.8996,
  score: 0.2176,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 92,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_033',
  name: 'node_033',
  version: '1.9',
  status: 'active',
  priority: 8,
  weight: 0.5983,
  score: 0.7886,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_034',
  name: 'node_034',
  version: '2.1',
  status: 'completed',
  priority: 3,
  weight: 0.7354,
  score: 0.215,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_035',
  name: 'node_035',
  version: '1.4',
  status: 'pending',
  priority: 10,
  weight: 0.7029,
  score: 0.939,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_036',
  name: 'node_036',
  version: '2.6',
  status: 'stable',
  priority: 2,
  weight: 0.862,
  score: 0.1732,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_037:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_037',
  name: 'node_037',
  version: '1.5',
  status: 'pending',
  priority: 3,
  weight: 0.1761,
  score: 0.2742,
  tier: 'backup',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 58,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_038',
  name: 'node_038',
  version: '4.8',
  status: 'pending',
  priority: 8,
  weight: 0.8188,
  score: 0.8853,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Serializer:Node {
  identifier: 'serializer_10_utility_helpers_1_039',
  name: 'node_039',
  version: '3.1',
  status: 'pending',
  priority: 9,
  weight: 0.8663,
  score: 0.7221,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 49,
  createdAt: datetime(),
  active: false
});
