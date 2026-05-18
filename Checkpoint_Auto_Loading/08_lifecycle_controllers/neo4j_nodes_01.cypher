:param namespace => 'checkpointloader_01_01';
:param batchSize => 256;
:param threshold => 0.52;
:param maxDepth => 4;
:param timeoutSeconds => 13;
:param region => 'eu-west';
:param epoch => 27;
:param version => '4.2.6';

CREATE (n_000:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_000',
  name: 'node_000',
  version: '5.5',
  status: 'recovered',
  priority: 1,
  weight: 0.2481,
  score: 0.6797,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 63,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_001',
  name: 'node_001',
  version: '1.0',
  status: 'failed',
  priority: 5,
  weight: 0.2258,
  score: 0.4548,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_1',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_002',
  name: 'node_002',
  version: '4.1',
  status: 'pending',
  priority: 6,
  weight: 0.638,
  score: 0.017,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_003',
  name: 'node_003',
  version: '5.3',
  status: 'pending',
  priority: 2,
  weight: 0.4659,
  score: 0.8139,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 74,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_004',
  name: 'node_004',
  version: '4.2',
  status: 'stable',
  priority: 2,
  weight: 0.1006,
  score: 0.9224,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 42,
  createdAt: datetime(),
  active: false
});

CREATE (n_005:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_005',
  name: 'node_005',
  version: '5.1',
  status: 'degraded',
  priority: 9,
  weight: 0.9452,
  score: 0.678,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_006:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_006',
  name: 'node_006',
  version: '5.5',
  status: 'failed',
  priority: 5,
  weight: 0.3499,
  score: 0.3526,
  tier: 'backup',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_007',
  name: 'node_007',
  version: '5.7',
  status: 'failed',
  priority: 9,
  weight: 0.4451,
  score: 0.3473,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_008',
  name: 'node_008',
  version: '1.7',
  status: 'pending',
  priority: 5,
  weight: 0.3865,
  score: 0.6207,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_009',
  name: 'node_009',
  version: '3.8',
  status: 'failed',
  priority: 10,
  weight: 0.4705,
  score: 0.2832,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'us-east',
  epoch: 17,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_010',
  name: 'node_010',
  version: '1.6',
  status: 'active',
  priority: 9,
  weight: 0.4663,
  score: 0.3871,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_011:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_011',
  name: 'node_011',
  version: '1.9',
  status: 'active',
  priority: 6,
  weight: 0.8904,
  score: 0.3572,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_012',
  name: 'node_012',
  version: '5.1',
  status: 'failed',
  priority: 3,
  weight: 0.5827,
  score: 0.6283,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_013',
  name: 'node_013',
  version: '2.1',
  status: 'failed',
  priority: 5,
  weight: 0.2574,
  score: 0.5404,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_014',
  name: 'node_014',
  version: '1.3',
  status: 'completed',
  priority: 10,
  weight: 0.1078,
  score: 0.4821,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 6,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_015',
  name: 'node_015',
  version: '3.9',
  status: 'active',
  priority: 10,
  weight: 0.7121,
  score: 0.9545,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'us-east',
  epoch: 59,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_016',
  name: 'node_016',
  version: '3.1',
  status: 'stable',
  priority: 1,
  weight: 0.4972,
  score: 0.7179,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_017',
  name: 'node_017',
  version: '3.6',
  status: 'failed',
  priority: 8,
  weight: 0.4288,
  score: 0.398,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_018',
  name: 'node_018',
  version: '2.9',
  status: 'failed',
  priority: 4,
  weight: 0.6226,
  score: 0.9084,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 41,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_019',
  name: 'node_019',
  version: '1.0',
  status: 'stable',
  priority: 10,
  weight: 0.2676,
  score: 0.6213,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_020',
  name: 'node_020',
  version: '1.9',
  status: 'stable',
  priority: 10,
  weight: 0.5113,
  score: 0.2315,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_021',
  name: 'node_021',
  version: '1.9',
  status: 'stable',
  priority: 2,
  weight: 0.8095,
  score: 0.8256,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 35,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_022',
  name: 'node_022',
  version: '4.4',
  status: 'active',
  priority: 10,
  weight: 0.1876,
  score: 0.0592,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_023:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_023',
  name: 'node_023',
  version: '3.2',
  status: 'degraded',
  priority: 3,
  weight: 0.7952,
  score: 0.0843,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 66,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_024',
  name: 'node_024',
  version: '2.4',
  status: 'degraded',
  priority: 8,
  weight: 0.1829,
  score: 0.3954,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'eu-west',
  epoch: 94,
  createdAt: datetime(),
  active: false
});

CREATE (n_025:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_025',
  name: 'node_025',
  version: '4.1',
  status: 'pending',
  priority: 2,
  weight: 0.1667,
  score: 0.3505,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 15,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_026',
  name: 'node_026',
  version: '1.6',
  status: 'degraded',
  priority: 10,
  weight: 0.1647,
  score: 0.7195,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_027',
  name: 'node_027',
  version: '2.1',
  status: 'active',
  priority: 3,
  weight: 0.7456,
  score: 0.7231,
  tier: 'edge',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 61,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_028',
  name: 'node_028',
  version: '4.9',
  status: 'active',
  priority: 9,
  weight: 0.3615,
  score: 0.915,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_029',
  name: 'node_029',
  version: '1.9',
  status: 'pending',
  priority: 10,
  weight: 0.8696,
  score: 0.6811,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 43,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_030',
  name: 'node_030',
  version: '5.9',
  status: 'failed',
  priority: 9,
  weight: 0.7749,
  score: 0.1248,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 57,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_031',
  name: 'node_031',
  version: '5.8',
  status: 'pending',
  priority: 9,
  weight: 0.8307,
  score: 0.6271,
  tier: 'backup',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_032',
  name: 'node_032',
  version: '1.0',
  status: 'stable',
  priority: 3,
  weight: 0.1001,
  score: 0.0017,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_033',
  name: 'node_033',
  version: '1.3',
  status: 'pending',
  priority: 2,
  weight: 0.622,
  score: 0.5095,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: true
});

CREATE (n_034:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_034',
  name: 'node_034',
  version: '4.2',
  status: 'stable',
  priority: 7,
  weight: 0.671,
  score: 0.8209,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 89,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_035',
  name: 'node_035',
  version: '5.6',
  status: 'stable',
  priority: 4,
  weight: 0.8136,
  score: 0.0452,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_036',
  name: 'node_036',
  version: '1.3',
  status: 'failed',
  priority: 7,
  weight: 0.1058,
  score: 0.1261,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 76,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_037',
  name: 'node_037',
  version: '5.8',
  status: 'failed',
  priority: 9,
  weight: 0.7657,
  score: 0.4579,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'ap-south',
  epoch: 81,
  createdAt: datetime(),
  active: false
});

CREATE (n_038:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_038',
  name: 'node_038',
  version: '4.1',
  status: 'active',
  priority: 5,
  weight: 0.6754,
  score: 0.2087,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:CheckpointLoader:Node {
  identifier: 'checkpointloader_08_lifecycle_controllers_1_039',
  name: 'node_039',
  version: '4.5',
  status: 'stable',
  priority: 9,
  weight: 0.4068,
  score: 0.0512,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 78,
  createdAt: datetime(),
  active: false
});
