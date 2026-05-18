:param namespace => 'basemodel_01_01';
:param batchSize => 128;
:param threshold => 0.518;
:param maxDepth => 10;
:param timeoutSeconds => 83;
:param region => 'eu-west';
:param epoch => 19;
:param version => '1.8.5';

CREATE (n_000:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_000',
  name: 'node_000',
  version: '5.3',
  status: 'recovered',
  priority: 3,
  weight: 0.832,
  score: 0.2386,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_0',
  region: 'ap-south',
  epoch: 59,
  createdAt: datetime(),
  active: true
});

CREATE (n_001:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_001',
  name: 'node_001',
  version: '1.5',
  status: 'failed',
  priority: 6,
  weight: 0.8489,
  score: 0.8178,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_1',
  region: 'us-east',
  epoch: 90,
  createdAt: datetime(),
  active: false
});

CREATE (n_002:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_002',
  name: 'node_002',
  version: '4.1',
  status: 'completed',
  priority: 6,
  weight: 0.6967,
  score: 0.5226,
  tier: 'tertiary',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: false
});

CREATE (n_003:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_003',
  name: 'node_003',
  version: '4.4',
  status: 'failed',
  priority: 2,
  weight: 0.5044,
  score: 0.4783,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 1,
  createdAt: datetime(),
  active: true
});

CREATE (n_004:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_004',
  name: 'node_004',
  version: '3.7',
  status: 'recovered',
  priority: 4,
  weight: 0.6605,
  score: 0.5234,
  tier: 'backup',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 72,
  createdAt: datetime(),
  active: true
});

CREATE (n_005:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_005',
  name: 'node_005',
  version: '1.9',
  status: 'completed',
  priority: 1,
  weight: 0.6315,
  score: 0.3065,
  tier: 'edge',
  mode: 'safe',
  category: 'category_0',
  region: 'eu-west',
  epoch: 33,
  createdAt: datetime(),
  active: true
});

CREATE (n_006:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_006',
  name: 'node_006',
  version: '3.7',
  status: 'active',
  priority: 9,
  weight: 0.6725,
  score: 0.8589,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'eu-west',
  epoch: 38,
  createdAt: datetime(),
  active: false
});

CREATE (n_007:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_007',
  name: 'node_007',
  version: '1.7',
  status: 'failed',
  priority: 6,
  weight: 0.1376,
  score: 0.7531,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_2',
  region: 'ap-south',
  epoch: 78,
  createdAt: datetime(),
  active: false
});

CREATE (n_008:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_008',
  name: 'node_008',
  version: '3.3',
  status: 'failed',
  priority: 10,
  weight: 0.2165,
  score: 0.6186,
  tier: 'edge',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 86,
  createdAt: datetime(),
  active: true
});

CREATE (n_009:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_009',
  name: 'node_009',
  version: '3.1',
  status: 'active',
  priority: 8,
  weight: 0.4415,
  score: 0.5258,
  tier: 'backup',
  mode: 'strict',
  category: 'category_4',
  region: 'us-east',
  epoch: 76,
  createdAt: datetime(),
  active: false
});

CREATE (n_010:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_010',
  name: 'node_010',
  version: '4.6',
  status: 'failed',
  priority: 8,
  weight: 0.2586,
  score: 0.0651,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_0',
  region: 'us-east',
  epoch: 66,
  createdAt: datetime(),
  active: true
});

CREATE (n_011:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_011',
  name: 'node_011',
  version: '2.3',
  status: 'failed',
  priority: 9,
  weight: 0.1365,
  score: 0.6798,
  tier: 'edge',
  mode: 'safe',
  category: 'category_1',
  region: 'eu-west',
  epoch: 99,
  createdAt: datetime(),
  active: false
});

CREATE (n_012:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_012',
  name: 'node_012',
  version: '1.1',
  status: 'pending',
  priority: 2,
  weight: 0.6139,
  score: 0.0155,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 73,
  createdAt: datetime(),
  active: false
});

CREATE (n_013:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_013',
  name: 'node_013',
  version: '1.3',
  status: 'degraded',
  priority: 6,
  weight: 0.5345,
  score: 0.0548,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_3',
  region: 'us-east',
  epoch: 53,
  createdAt: datetime(),
  active: true
});

CREATE (n_014:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_014',
  name: 'node_014',
  version: '2.5',
  status: 'completed',
  priority: 4,
  weight: 0.5664,
  score: 0.006,
  tier: 'edge',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 34,
  createdAt: datetime(),
  active: true
});

CREATE (n_015:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_015',
  name: 'node_015',
  version: '3.6',
  status: 'completed',
  priority: 5,
  weight: 0.6001,
  score: 0.511,
  tier: 'backup',
  mode: 'strict',
  category: 'category_0',
  region: 'eu-west',
  epoch: 39,
  createdAt: datetime(),
  active: true
});

CREATE (n_016:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_016',
  name: 'node_016',
  version: '4.6',
  status: 'stable',
  priority: 9,
  weight: 0.3314,
  score: 0.202,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 84,
  createdAt: datetime(),
  active: false
});

CREATE (n_017:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_017',
  name: 'node_017',
  version: '4.7',
  status: 'degraded',
  priority: 10,
  weight: 0.2272,
  score: 0.9305,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_2',
  region: 'eu-west',
  epoch: 91,
  createdAt: datetime(),
  active: true
});

CREATE (n_018:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_018',
  name: 'node_018',
  version: '3.0',
  status: 'recovered',
  priority: 2,
  weight: 0.468,
  score: 0.5649,
  tier: 'tertiary',
  mode: 'strict',
  category: 'category_3',
  region: 'eu-west',
  epoch: 29,
  createdAt: datetime(),
  active: false
});

CREATE (n_019:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_019',
  name: 'node_019',
  version: '3.3',
  status: 'degraded',
  priority: 4,
  weight: 0.8222,
  score: 0.5921,
  tier: 'backup',
  mode: 'experimental',
  category: 'category_4',
  region: 'ap-south',
  epoch: 57,
  createdAt: datetime(),
  active: true
});

CREATE (n_020:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_020',
  name: 'node_020',
  version: '2.0',
  status: 'pending',
  priority: 7,
  weight: 0.8723,
  score: 0.1245,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_0',
  region: 'ap-south',
  epoch: 64,
  createdAt: datetime(),
  active: true
});

CREATE (n_021:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_021',
  name: 'node_021',
  version: '1.8',
  status: 'degraded',
  priority: 3,
  weight: 0.5484,
  score: 0.6739,
  tier: 'tertiary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'ap-south',
  epoch: 21,
  createdAt: datetime(),
  active: true
});

CREATE (n_022:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_022',
  name: 'node_022',
  version: '2.8',
  status: 'active',
  priority: 8,
  weight: 0.1857,
  score: 0.7843,
  tier: 'primary',
  mode: 'experimental',
  category: 'category_2',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: false
});

CREATE (n_023:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_023',
  name: 'node_023',
  version: '4.6',
  status: 'pending',
  priority: 1,
  weight: 0.9315,
  score: 0.1334,
  tier: 'secondary',
  mode: 'experimental',
  category: 'category_3',
  region: 'eu-west',
  epoch: 98,
  createdAt: datetime(),
  active: true
});

CREATE (n_024:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_024',
  name: 'node_024',
  version: '5.5',
  status: 'degraded',
  priority: 9,
  weight: 0.7474,
  score: 0.3096,
  tier: 'tertiary',
  mode: 'safe',
  category: 'category_4',
  region: 'ap-south',
  epoch: 28,
  createdAt: datetime(),
  active: true
});

CREATE (n_025:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_025',
  name: 'node_025',
  version: '2.6',
  status: 'active',
  priority: 6,
  weight: 0.442,
  score: 0.6408,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 12,
  createdAt: datetime(),
  active: true
});

CREATE (n_026:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_026',
  name: 'node_026',
  version: '4.2',
  status: 'degraded',
  priority: 3,
  weight: 0.4869,
  score: 0.6789,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'eu-west',
  epoch: 16,
  createdAt: datetime(),
  active: true
});

CREATE (n_027:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_027',
  name: 'node_027',
  version: '5.8',
  status: 'active',
  priority: 5,
  weight: 0.5409,
  score: 0.0178,
  tier: 'backup',
  mode: 'strict',
  category: 'category_2',
  region: 'us-east',
  epoch: 63,
  createdAt: datetime(),
  active: false
});

CREATE (n_028:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_028',
  name: 'node_028',
  version: '3.9',
  status: 'recovered',
  priority: 9,
  weight: 0.7806,
  score: 0.2013,
  tier: 'backup',
  mode: 'safe',
  category: 'category_3',
  region: 'us-east',
  epoch: 75,
  createdAt: datetime(),
  active: false
});

CREATE (n_029:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_029',
  name: 'node_029',
  version: '1.9',
  status: 'recovered',
  priority: 2,
  weight: 0.9706,
  score: 0.3443,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_4',
  region: 'us-east',
  epoch: 23,
  createdAt: datetime(),
  active: false
});

CREATE (n_030:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_030',
  name: 'node_030',
  version: '3.7',
  status: 'failed',
  priority: 4,
  weight: 0.3966,
  score: 0.3641,
  tier: 'primary',
  mode: 'safe',
  category: 'category_0',
  region: 'us-east',
  epoch: 93,
  createdAt: datetime(),
  active: false
});

CREATE (n_031:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_031',
  name: 'node_031',
  version: '1.8',
  status: 'active',
  priority: 3,
  weight: 0.636,
  score: 0.4614,
  tier: 'primary',
  mode: 'strict',
  category: 'category_1',
  region: 'ap-south',
  epoch: 75,
  createdAt: datetime(),
  active: true
});

CREATE (n_032:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_032',
  name: 'node_032',
  version: '4.2',
  status: 'failed',
  priority: 10,
  weight: 0.8535,
  score: 0.0762,
  tier: 'secondary',
  mode: 'safe',
  category: 'category_2',
  region: 'us-east',
  epoch: 85,
  createdAt: datetime(),
  active: true
});

CREATE (n_033:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_033',
  name: 'node_033',
  version: '3.0',
  status: 'stable',
  priority: 8,
  weight: 0.373,
  score: 0.2613,
  tier: 'primary',
  mode: 'relaxed',
  category: 'category_3',
  region: 'us-east',
  epoch: 20,
  createdAt: datetime(),
  active: false
});

CREATE (n_034:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_034',
  name: 'node_034',
  version: '3.8',
  status: 'recovered',
  priority: 2,
  weight: 0.3918,
  score: 0.246,
  tier: 'edge',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 65,
  createdAt: datetime(),
  active: false
});

CREATE (n_035:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_035',
  name: 'node_035',
  version: '3.3',
  status: 'completed',
  priority: 7,
  weight: 0.5998,
  score: 0.988,
  tier: 'secondary',
  mode: 'shadow',
  category: 'category_0',
  region: 'ap-south',
  epoch: 31,
  createdAt: datetime(),
  active: true
});

CREATE (n_036:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_036',
  name: 'node_036',
  version: '1.1',
  status: 'active',
  priority: 8,
  weight: 0.8125,
  score: 0.7014,
  tier: 'secondary',
  mode: 'relaxed',
  category: 'category_1',
  region: 'us-east',
  epoch: 97,
  createdAt: datetime(),
  active: true
});

CREATE (n_037:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_037',
  name: 'node_037',
  version: '2.4',
  status: 'active',
  priority: 7,
  weight: 0.4539,
  score: 0.5181,
  tier: 'tertiary',
  mode: 'shadow',
  category: 'category_2',
  region: 'us-east',
  epoch: 11,
  createdAt: datetime(),
  active: true
});

CREATE (n_038:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_038',
  name: 'node_038',
  version: '2.3',
  status: 'recovered',
  priority: 9,
  weight: 0.7397,
  score: 0.0621,
  tier: 'secondary',
  mode: 'strict',
  category: 'category_3',
  region: 'ap-south',
  epoch: 44,
  createdAt: datetime(),
  active: true
});

CREATE (n_039:BaseModel:Node {
  identifier: 'basemodel_05_metric_trackers_1_039',
  name: 'node_039',
  version: '1.3',
  status: 'recovered',
  priority: 3,
  weight: 0.8332,
  score: 0.3421,
  tier: 'backup',
  mode: 'shadow',
  category: 'category_4',
  region: 'us-east',
  epoch: 2,
  createdAt: datetime(),
  active: false
});
