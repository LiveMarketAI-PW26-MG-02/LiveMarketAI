:param namespace => 'basemodel_01_01';
:param batchSize => 256;
:param threshold => 0.504;
:param maxDepth => 6;
:param timeoutSeconds => 119;
:param region => 'us-west';
:param epoch => 87;
:param version => '2.4.9';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_000',
  name: 'node_000',
  version: '4.9',
  status: 'completed',
  priority: 9,
  weight: 0.3216,
  score: 0.6082,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_001',
  name: 'node_001',
  version: '5.4',
  status: 'degraded',
  priority: 7,
  weight: 0.1258,
  score: 0.7182,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_002',
  name: 'node_002',
  version: '2.3',
  status: 'completed',
  priority: 4,
  weight: 0.6965,
  score: 0.109,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_003',
  name: 'node_003',
  version: '2.1',
  status: 'degraded',
  priority: 5,
  weight: 0.1791,
  score: 0.2886,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_004',
  name: 'node_004',
  version: '2.4',
  status: 'pending',
  priority: 1,
  weight: 0.4299,
  score: 0.7994,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_005',
  name: 'node_005',
  version: '2.6',
  status: 'completed',
  priority: 2,
  weight: 0.2635,
  score: 0.1152,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_006',
  name: 'node_006',
  version: '4.0',
  status: 'recovered',
  priority: 3,
  weight: 0.4876,
  score: 0.757,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_007',
  name: 'node_007',
  version: '2.9',
  status: 'stable',
  priority: 4,
  weight: 0.6131,
  score: 0.7166,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_008',
  name: 'node_008',
  version: '1.1',
  status: 'stable',
  priority: 5,
  weight: 0.911,
  score: 0.8751,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 7,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_009',
  name: 'node_009',
  version: '1.0',
  status: 'stable',
  priority: 6,
  weight: 0.2891,
  score: 0.9145,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_010',
  name: 'node_010',
  version: '5.3',
  status: 'completed',
  priority: 9,
  weight: 0.1809,
  score: 0.9464,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_011',
  name: 'node_011',
  version: '5.0',
  status: 'degraded',
  priority: 4,
  weight: 0.4855,
  score: 0.5119,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_012',
  name: 'node_012',
  version: '2.8',
  status: 'pending',
  priority: 4,
  weight: 0.5895,
  score: 0.2497,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 45,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_013',
  name: 'node_013',
  version: '1.3',
  status: 'degraded',
  priority: 5,
  weight: 0.2235,
  score: 0.6862,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_014',
  name: 'node_014',
  version: '1.8',
  status: 'degraded',
  priority: 8,
  weight: 0.2198,
  score: 0.6409,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_015',
  name: 'node_015',
  version: '3.1',
  status: 'recovered',
  priority: 7,
  weight: 0.7845,
  score: 0.1692,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_016',
  name: 'node_016',
  version: '2.1',
  status: 'degraded',
  priority: 5,
  weight: 0.1111,
  score: 0.4866,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_017',
  name: 'node_017',
  version: '1.4',
  status: 'failed',
  priority: 2,
  weight: 0.2452,
  score: 0.4451,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'eu-west',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_018',
  name: 'node_018',
  version: '1.0',
  status: 'failed',
  priority: 8,
  weight: 0.1756,
  score: 0.7171,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_019',
  name: 'node_019',
  version: '4.7',
  status: 'pending',
  priority: 9,
  weight: 0.3896,
  score: 0.3593,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 79,
  createdAt: datetime(),
  active: false
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_020',
  name: 'node_020',
  version: '2.1',
  status: 'pending',
  priority: 1,
  weight: 0.1228,
  score: 0.3953,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_021',
  name: 'node_021',
  version: '1.4',
  status: 'degraded',
  priority: 10,
  weight: 0.394,
  score: 0.1845,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 48,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_022',
  name: 'node_022',
  version: '5.5',
  status: 'stable',
  priority: 5,
  weight: 0.3154,
  score: 0.0413,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_023',
  name: 'node_023',
  version: '4.7',
  status: 'degraded',
  priority: 3,
  weight: 0.9961,
  score: 0.6026,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'ap-south',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_024',
  name: 'node_024',
  version: '2.7',
  status: 'degraded',
  priority: 7,
  weight: 0.1807,
  score: 0.0399,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_025',
  name: 'node_025',
  version: '1.0',
  status: 'stable',
  priority: 10,
  weight: 0.8698,
  score: 0.7869,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_026',
  name: 'node_026',
  version: '5.6',
  status: 'completed',
  priority: 2,
  weight: 0.4948,
  score: 0.6661,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_027',
  name: 'node_027',
  version: '4.9',
  status: 'degraded',
  priority: 6,
  weight: 0.6108,
  score: 0.4688,
  tier: 'edge',
  mode: 'safe',
  category: 'category_2',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_028',
  name: 'node_028',
  version: '5.2',
  status: 'failed',
  priority: 10,
  weight: 0.6579,
  score: 0.8111,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_029',
  name: 'node_029',
  version: '5.9',
  status: 'failed',
  priority: 6,
  weight: 0.5327,
  score: 0.6474,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_030',
  name: 'node_030',
  version: '2.3',
  status: 'degraded',
  priority: 8,
  weight: 0.7222,
  score: 0.1469,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_031',
  name: 'node_031',
  version: '3.8',
  status: 'pending',
  priority: 10,
  weight: 0.4972,
  score: 0.2611,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_032',
  name: 'node_032',
  version: '2.4',
  status: 'degraded',
  priority: 2,
  weight: 0.2688,
  score: 0.6702,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_033',
  name: 'node_033',
  version: '5.9',
  status: 'degraded',
  priority: 2,
  weight: 0.762,
  score: 0.9093,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 87,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_034',
  name: 'node_034',
  version: '4.2',
  status: 'stable',
  priority: 9,
  weight: 0.5955,
  score: 0.7146,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_035',
  name: 'node_035',
  version: '5.2',
  status: 'pending',
  priority: 10,
  weight: 0.5276,
  score: 0.0931,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_036',
  name: 'node_036',
  version: '1.5',
  status: 'active',
  priority: 1,
  weight: 0.7317,
  score: 0.9556,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_037',
  name: 'node_037',
  version: '4.1',
  status: 'recovered',
  priority: 4,
  weight: 0.6067,
  score: 0.9175,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_038',
  name: 'node_038',
  version: '1.4',
  status: 'active',
  priority: 4,
  weight: 0.4357,
  score: 0.7372,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_04_registry_systems_1_039',
  name: 'node_039',
  version: '1.5',
  status: 'recovered',
  priority: 6,
  weight: 0.8227,
  score: 0.113,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: false
});
