:param namespace => 'checkpointloader_01_01';
:param batchSize => 512;
:param threshold => 0.75;
:param maxDepth => 11;
:param timeoutSeconds => 110;
:param region => 'us-east';
:param epoch => 48;
:param version => '2.7.0';

CREATE (n_000:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_000',
  name: 'node_000',
  version: '4.6',
  status: 'stable',
  priority: 3,
  weight: 0.486,
  score: 0.851,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'ap-south',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_001',
  name: 'node_001',
  version: '5.8',
  status: 'completed',
  priority: 6,
  weight: 0.6947,
  score: 0.1313,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 40,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_002',
  name: 'node_002',
  version: '3.4',
  status: 'failed',
  priority: 8,
  weight: 0.8859,
  score: 0.0051,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'eu-west',
  epoch: 77,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_003',
  name: 'node_003',
  version: '2.7',
  status: 'failed',
  priority: 10,
  weight: 0.4189,
  score: 0.4954,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_004:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_004',
  name: 'node_004',
  version: '3.5',
  status: 'active',
  priority: 7,
  weight: 0.1435,
  score: 0.2965,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_005',
  name: 'node_005',
  version: '5.0',
  status: 'completed',
  priority: 8,
  weight: 0.4066,
  score: 0.268,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 83,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_006',
  name: 'node_006',
  version: '5.0',
  status: 'completed',
  priority: 6,
  weight: 0.1323,
  score: 0.596,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_007',
  name: 'node_007',
  version: '2.6',
  status: 'failed',
  priority: 8,
  weight: 0.8911,
  score: 0.3052,
  tier: 'primary',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_008:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_008',
  name: 'node_008',
  version: '3.1',
  status: 'degraded',
  priority: 9,
  weight: 0.1884,
  score: 0.837,
  tier: 'edge',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_009',
  name: 'node_009',
  version: '1.8',
  status: 'completed',
  priority: 2,
  weight: 0.5798,
  score: 0.7325,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 28,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_010',
  name: 'node_010',
  version: '1.7',
  status: 'completed',
  priority: 1,
  weight: 0.8934,
  score: 0.5205,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 52,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_011',
  name: 'node_011',
  version: '1.4',
  status: 'failed',
  priority: 3,
  weight: 0.2317,
  score: 0.8525,
  tier: 'primary',
  mode: 'safe',
  category: 'category_1',
  region: 'us-east',
  epoch: 55,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_012',
  name: 'node_012',
  version: '2.7',
  status: 'active',
  priority: 10,
  weight: 0.1611,
  score: 0.5861,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'ap-south',
  epoch: 26,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_013',
  name: 'node_013',
  version: '4.8',
  status: 'degraded',
  priority: 7,
  weight: 0.4836,
  score: 0.7804,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_3',
  region: 'us-east',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_014:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_014',
  name: 'node_014',
  version: '3.0',
  status: 'degraded',
  priority: 4,
  weight: 0.1409,
  score: 0.6818,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_4',
  region: 'eu-west',
  epoch: 8,
  createdAt: datetime(),
  active: false
});

CREATE (n_015:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_015',
  name: 'node_015',
  version: '2.2',
  status: 'failed',
  priority: 10,
  weight: 0.9854,
  score: 0.3648,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 72,
  createdAt: datetime(),
  active: false
});

CREATE (n_016:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_016',
  name: 'node_016',
  version: '3.0',
  status: 'stable',
  priority: 1,
  weight: 0.2489,
  score: 0.2794,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_017',
  name: 'node_017',
  version: '1.3',
  status: 'pending',
  priority: 3,
  weight: 0.5514,
  score: 0.2752,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_018',
  name: 'node_018',
  version: '2.6',
  status: 'stable',
  priority: 3,
  weight: 0.6476,
  score: 0.9137,
  tier: 'primary',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 24,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_019',
  name: 'node_019',
  version: '3.7',
  status: 'recovered',
  priority: 1,
  weight: 0.2261,
  score: 0.4511,
  tier: 'primary',
  mode: 'safe',
  category: 'category_4',
  region: 'eu-west',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_020',
  name: 'node_020',
  version: '4.7',
  status: 'recovered',
  priority: 10,
  weight: 0.2379,
  score: 0.5011,
  tier: 'edge',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: false
});

CREATE (n_021:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_021',
  name: 'node_021',
  version: '3.1',
  status: 'recovered',
  priority: 8,
  weight: 0.7914,
  score: 0.7871,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 88,
  createdAt: datetime(),
  active: false
});

CREATE (n_022:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_022',
  name: 'node_022',
  version: '4.2',
  status: 'completed',
  priority: 7,
  weight: 0.9111,
  score: 0.1752,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 50,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_023',
  name: 'node_023',
  version: '4.7',
  status: 'pending',
  priority: 2,
  weight: 0.1556,
  score: 0.1401,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 96,
  createdAt: datetime(),
  active: false
});

CREATE (n_024:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_024',
  name: 'node_024',
  version: '4.4',
  status: 'failed',
  priority: 3,
  weight: 0.3122,
  score: 0.6369,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'eu-west',
  epoch: 82,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_025',
  name: 'node_025',
  version: '5.7',
  status: 'stable',
  priority: 7,
  weight: 0.5271,
  score: 0.5114,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_026',
  name: 'node_026',
  version: '4.0',
  status: 'degraded',
  priority: 9,
  weight: 0.924,
  score: 0.0369,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 19,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_027',
  name: 'node_027',
  version: '3.4',
  status: 'pending',
  priority: 2,
  weight: 0.5963,
  score: 0.1563,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_028:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_028',
  name: 'node_028',
  version: '4.2',
  status: 'completed',
  priority: 2,
  weight: 0.256,
  score: 0.7458,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 51,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_029',
  name: 'node_029',
  version: '4.3',
  status: 'degraded',
  priority: 1,
  weight: 0.8456,
  score: 0.3245,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_030',
  name: 'node_030',
  version: '5.2',
  status: 'pending',
  priority: 9,
  weight: 0.5742,
  score: 0.6429,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'us-east',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_031',
  name: 'node_031',
  version: '5.7',
  status: 'failed',
  priority: 6,
  weight: 0.9625,
  score: 0.7547,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_1',
  region: 'ap-south',
  epoch: 23,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_032',
  name: 'node_032',
  version: '1.3',
  status: 'stable',
  priority: 10,
  weight: 0.3542,
  score: 0.7434,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'us-east',
  epoch: 54,
  createdAt: datetime(),
  active: false
});

CREATE (n_033:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_033',
  name: 'node_033',
  version: '5.8',
  status: 'failed',
  priority: 6,
  weight: 0.2759,
  score: 0.7497,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_034',
  name: 'node_034',
  version: '2.2',
  status: 'pending',
  priority: 8,
  weight: 0.648,
  score: 0.0766,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'ap-south',
  epoch: 2,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_035',
  name: 'node_035',
  version: '1.5',
  status: 'failed',
  priority: 1,
  weight: 0.3323,
  score: 0.3489,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_036:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_036',
  name: 'node_036',
  version: '2.2',
  status: 'active',
  priority: 2,
  weight: 0.1573,
  score: 0.6393,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 19,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_037',
  name: 'node_037',
  version: '4.8',
  status: 'active',
  priority: 9,
  weight: 0.6793,
  score: 0.7079,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_038',
  name: 'node_038',
  version: '5.5',
  status: 'stable',
  priority: 2,
  weight: 0.6486,
  score: 0.3191,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'eu-west',
  epoch: 48,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:CheckpointLoader:Node {
  identifier: 'checkpointloader_02_state_handlers_1_039',
  name: 'node_039',
  version: '1.5',
  status: 'degraded',
  priority: 6,
  weight: 0.7706,
  score: 0.9891,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 95,
  createdAt: datetime(),
  active: false
});
