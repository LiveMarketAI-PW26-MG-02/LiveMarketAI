:param namespace => 'inferencecontext_04_04';
:param batchSize => 32;
:param threshold => 0.328;
:param maxDepth => 11;
:param timeoutSeconds => 16;
:param region => 'eu-west';
:param epoch => 76;
:param version => '3.0.0';

CREATE INDEX idx_inferencecontext_02_state_handlers_4_identifier_00
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.identifier);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_name_01
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.name);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_version_02
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.version);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_status_03
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.status);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_priority_04
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.priority);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_category_05
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.category);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_weight_06
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.weight);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_score_07
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.score);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_confidence_08
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.confidence);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_latency_09
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.latency);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_throughput_10
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.throughput);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_accuracy_11
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.accuracy);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_owner_12
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.owner);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_namespace_13
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.namespace);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_region_14
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.region);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_tier_15
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.tier);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_phase_16
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.phase);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_mode_17
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.mode);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_kind_18
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.kind);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_checksum_19
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.checksum);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_epoch_20
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.epoch);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_iteration_21
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.iteration);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_depth_22
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.depth);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_width_23
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.width);

CREATE INDEX idx_inferencecontext_02_state_handlers_4_rank_24
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.rank);

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_unique_identifier_0
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.identifier IS UNIQUE;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_unique_name_1
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.name IS UNIQUE;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_unique_checksum_2
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.checksum IS UNIQUE;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_unique_namespace_3
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.namespace IS UNIQUE;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_status_0
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.status IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_priority_1
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.priority IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_weight_2
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.weight IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_version_3
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.version IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_tier_4
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.tier IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_mode_5
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.mode IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_category_6
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.category IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_region_7
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.region IS NOT NULL;

CREATE CONSTRAINT constraint_inferencecontext_02_state_handlers_4_exists_epoch_8
IF NOT EXISTS
FOR (n:InferenceContext)
REQUIRE n.epoch IS NOT NULL;

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_00
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.namespace, n.epoch);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_01
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.status, n.priority);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_02
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.region, n.tier);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_03
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.category, n.mode);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_04
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.version, n.status);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_05
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.owner, n.namespace);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_06
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.checksum, n.version);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_07
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.priority, n.score);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_08
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.score, n.confidence);

CREATE INDEX composite_idx_inferencecontext_02_state_handlers_4_09
IF NOT EXISTS
FOR (n:InferenceContext)
ON (n.tier, n.mode);

CREATE INDEX filler_idx_0249 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0250 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0251 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0252 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0253 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0254 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0255 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0256 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0257 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0258 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0259 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0260 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0261 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0262 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0263 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0264 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0265 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0266 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0267 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0268 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0269 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0270 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0271 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0272 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0273 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0274 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0275 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0276 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0277 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0278 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0279 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0280 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0281 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0282 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0283 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0284 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0285 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0286 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0287 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0288 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0289 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0290 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0291 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0292 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0293 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0294 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0295 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0296 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0297 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0298 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0299 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0300 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0301 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0302 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0303 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0304 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0305 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0306 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0307 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0308 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0309 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0310 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0311 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0312 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0313 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0314 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0315 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0316 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0317 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0318 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);
CREATE INDEX filler_idx_0319 IF NOT EXISTS FOR (n:InferenceContext) ON (n.score);