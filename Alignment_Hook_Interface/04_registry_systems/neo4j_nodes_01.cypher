:param namespace => 'alignment_01_01';
:param batchSize => 256;
:param threshold => 0.767;
:param maxDepth => 9;
:param timeoutSeconds => 44;
:param region => 'us-east';
:param epoch => 45;
:param version => '2.5.5';

CREATE (n_000:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_000',
  name: 'node_000',
  version: '1.8',
  status: 'active',
  priority: 3,
  weight: 0.3868,
  score: 0.0511,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 46,
  createdAt: datetime(),
  active: false
});

CREATE (n_001:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_001',
  name: 'node_001',
  version: '3.6',
  status: 'completed',
  priority: 4,
  weight: 0.3488,
  score: 0.2213,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_002:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_002',
  name: 'node_002',
  version: '4.7',
  status: 'completed',
  priority: 1,
  weight: 0.7865,
  score: 0.5453,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_003:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_003',
  name: 'node_003',
  version: '1.9',
  status: 'degraded',
  priority: 3,
  weight: 0.359,
  score: 0.8631,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_004',
  name: 'node_004',
  version: '3.2',
  status: 'pending',
  priority: 7,
  weight: 0.5582,
  score: 0.7367,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'ap-south',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_005',
  name: 'node_005',
  version: '3.2',
  status: 'stable',
  priority: 2,
  weight: 0.7772,
  score: 0.6376,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 95,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_006',
  name: 'node_006',
  version: '5.1',
  status: 'pending',
  priority: 8,
  weight: 0.2449,
  score: 0.4568,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 100,
  createdAt: datetime(),
  active: true
});

CREATE (n_007:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_007',
  name: 'node_007',
  version: '4.7',
  status: 'completed',
  priority: 4,
  weight: 0.8878,
  score: 0.1822,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_008',
  name: 'node_008',
  version: '5.7',
  status: 'active',
  priority: 2,
  weight: 0.3201,
  score: 0.731,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 56,
  createdAt: datetime(),
  active: false
});

CREATE (n_009:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_009',
  name: 'node_009',
  version: '2.5',
  status: 'pending',
  priority: 6,
  weight: 0.2595,
  score: 0.9571,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_010',
  name: 'node_010',
  version: '1.5',
  status: 'degraded',
  priority: 6,
  weight: 0.4753,
  score: 0.4881,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'eu-west',
  epoch: 70,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_011',
  name: 'node_011',
  version: '5.2',
  status: 'stable',
  priority: 1,
  weight: 0.55,
  score: 0.2374,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 36,
  createdAt: datetime(),
  active: true
});

CREATE (n_012:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_012',
  name: 'node_012',
  version: '1.2',
  status: 'completed',
  priority: 6,
  weight: 0.9688,
  score: 0.668,
  tier: 'edge',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 2,
  createdAt: datetime(),
  active: true
});

CREATE (n_013:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_013',
  name: 'node_013',
  version: '5.9',
  status: 'pending',
  priority: 10,
  weight: 0.8041,
  score: 0.9496,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_3',
  region: 'eu-west',
  epoch: 10,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_014',
  name: 'node_014',
  version: '5.0',
  status: 'degraded',
  priority: 1,
  weight: 0.9795,
  score: 0.153,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_015',
  name: 'node_015',
  version: '1.4',
  status: 'degraded',
  priority: 4,
  weight: 0.1866,
  score: 0.8569,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 3,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_016',
  name: 'node_016',
  version: '3.0',
  status: 'pending',
  priority: 10,
  weight: 0.5298,
  score: 0.4278,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 30,
  createdAt: datetime(),
  active: true
});

CREATE (n_017:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_017',
  name: 'node_017',
  version: '1.0',
  status: 'active',
  priority: 7,
  weight: 0.334,
  score: 0.374,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_2',
  region: 'ap-south',
  epoch: 6,
  createdAt: datetime(),
  active: false
});

CREATE (n_018:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_018',
  name: 'node_018',
  version: '1.7',
  status: 'pending',
  priority: 2,
  weight: 0.7318,
  score: 0.8338,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'ap-south',
  epoch: 69,
  createdAt: datetime(),
  active: true
});

CREATE (n_019:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_019',
  name: 'node_019',
  version: '1.2',
  status: 'completed',
  priority: 5,
  weight: 0.3656,
  score: 0.1488,
  tier: 'primary',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 67,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_020',
  name: 'node_020',
  version: '5.0',
  status: 'recovered',
  priority: 7,
  weight: 0.7606,
  score: 0.8291,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_0',
  region: 'eu-west',
  epoch: 81,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_021',
  name: 'node_021',
  version: '5.4',
  status: 'stable',
  priority: 3,
  weight: 0.8213,
  score: 0.4928,
  tier: 'edge',
  mode: 'strict',
  category: 'category_1',
  region: 'us-east',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_022',
  name: 'node_022',
  version: '5.1',
  status: 'failed',
  priority: 4,
  weight: 0.2217,
  score: 0.1685,
  tier: 'edge',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_023',
  name: 'node_023',
  version: '5.5',
  status: 'degraded',
  priority: 1,
  weight: 0.5872,
  score: 0.1742,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'ap-south',
  epoch: 71,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_024',
  name: 'node_024',
  version: '5.0',
  status: 'completed',
  priority: 6,
  weight: 0.6132,
  score: 0.0353,
  tier: 'primary',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 22,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_025',
  name: 'node_025',
  version: '4.2',
  status: 'recovered',
  priority: 5,
  weight: 0.6019,
  score: 0.6556,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_026',
  name: 'node_026',
  version: '1.8',
  status: 'pending',
  priority: 1,
  weight: 0.82,
  score: 0.8117,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_1',
  region: 'ap-south',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_027:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_027',
  name: 'node_027',
  version: '2.7',
  status: 'degraded',
  priority: 1,
  weight: 0.4066,
  score: 0.7866,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_028',
  name: 'node_028',
  version: '4.6',
  status: 'failed',
  priority: 8,
  weight: 0.345,
  score: 0.1707,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 29,
  createdAt: datetime(),
  active: true
});

CREATE (n_029:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_029',
  name: 'node_029',
  version: '4.3',
  status: 'failed',
  priority: 10,
  weight: 0.9827,
  score: 0.9508,
  tier: 'edge',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 27,
  createdAt: datetime(),
  active: true
});

CREATE (n_030:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_030',
  name: 'node_030',
  version: '1.0',
  status: 'recovered',
  priority: 10,
  weight: 0.4725,
  score: 0.8039,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_0',
  region: 'eu-west',
  epoch: 68,
  createdAt: datetime(),
  active: true
});

CREATE (n_031:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_031',
  name: 'node_031',
  version: '1.8',
  status: 'recovered',
  priority: 10,
  weight: 0.6536,
  score: 0.0843,
  tier: 'backup',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 24,
  createdAt: datetime(),
  active: false
});

CREATE (n_032:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_032',
  name: 'node_032',
  version: '2.2',
  status: 'completed',
  priority: 2,
  weight: 0.3071,
  score: 0.1629,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_2',
  region: 'eu-west',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_033',
  name: 'node_033',
  version: '3.1',
  status: 'stable',
  priority: 9,
  weight: 0.543,
  score: 0.86,
  tier: 'backup',
  mode: 'strict',
  category: 'category_3',
  region: 'us-east',
  epoch: 60,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_034',
  name: 'node_034',
  version: '4.8',
  status: 'recovered',
  priority: 3,
  weight: 0.3834,
  score: 0.2789,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_4',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_035',
  name: 'node_035',
  version: '1.4',
  status: 'completed',
  priority: 6,
  weight: 0.5443,
  score: 0.355,
  tier: 'backup',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 32,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_036',
  name: 'node_036',
  version: '4.2',
  status: 'stable',
  priority: 3,
  weight: 0.6344,
  score: 0.4747,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_037',
  name: 'node_037',
  version: '5.6',
  status: 'recovered',
  priority: 3,
  weight: 0.583,
  score: 0.7304,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_038',
  name: 'node_038',
  version: '5.3',
  status: 'failed',
  priority: 3,
  weight: 0.3453,
  score: 0.0882,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 25,
  createdAt: datetime(),
  active: false
});

CREATE (n_039:Alignment:Node {
  identifier: 'alignment_04_registry_systems_1_039',
  name: 'node_039',
  version: '1.5',
  status: 'recovered',
  priority: 1,
  weight: 0.437,
  score: 0.482,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_4',
  region: 'ap-south',
  epoch: 97,
  createdAt: datetime(),
  active: false
});
