export type JobType =
  | 'system_probe'
  | 'research_sync'
  | 'data_ingest'
  | 'data_build'
  | 'data_remove'
  | 'data_audit'
  | 'tokenizer_train'
  | 'tokenizer_shard'
  | 'tokenizer_evaluate'
  | 'kernel_train'
  | 'kernel_evaluate'
  | 'eval_suite'
  | 'eval_contamination'
  | 'scale_profile'
  | 'scale_sweep'
  | 'scale_mixture'
  | 'align_generate'
  | 'align_judge'
  | 'align_dataset'
  | 'align_sft'
  | 'align_preference'
  | 'align_evaluate'
  | 'tiny_train'
  | 'tiny_eval'
export type JobStatus =
  | 'queued'
  | 'starting'
  | 'running'
  | 'cancelling'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

export interface Health {
  status: 'ok'
  database: 'ok'
  version: string
}

export interface Capabilities {
  bind_host: string
  allowed_origin: string
  job_types: JobType[]
  cancellation: boolean
  event_transport: 'polling'
  arbitrary_commands: false
  arbitrary_url_fetching: false
  filesystem_browser: false
  hardware: Record<string, unknown>
}

export interface ResearchSource {
  id: string
  title: string
  organization: string
  publication_date: string
  media_type: string
  canonical_url: string
  tags: string[]
  evidence_classes: string[]
  integrity: 'pinned' | 'content-pinned' | 'unpinned'
  sync_status: 'verified' | 'missing' | 'error' | 'unknown'
  raw_sha256: string | null
  tracked_raw_sha256: string | null
  tracked_content_sha256: string | null
  content_scope: 'document' | 'article' | null
  content_profile: 'html-document-text-v1' | 'html-article-text-v1' | null
  size_bytes: number | null
  status_message: string
}

export interface ResearchCatalog {
  schema_version: number | null
  available: boolean
  catalog_path: string
  sources: ResearchSource[]
  count: number
  error: string | null
}

export interface DataSource {
  id: string
  title: string
  owner_or_steward: string
  license_id: string
  allowed_uses: string[]
  sensitive_content_risk: string
}

export interface DataSummary {
  schema_version: number
  registrations: number
  documents: number
  admitted: number
  quarantined: number
  tombstones: number
  builds: number
  integrity_errors: string[]
  latest_builds: Record<string, string>
  sources: DataSource[]
}

export interface TokenizerArtifact {
  tokenizer_id: string
  tokenizer_version: string
  reserved_spec_version: string
  declared_vocab_size: number
  learned_vocab_size: number
  merge_count: number
  dataset_id: string
  build_id: string
  train_split_sha256: string
  train_documents: number
  train_bytes: number
  min_frequency: number
  library_version: string
}

export interface TokenizedSplit {
  document_count: number
  excluded_document_count: number
  token_count: number
  shard_count: number
}

export interface TokenizedBuild {
  tokenized_id: string
  tokenizer_id: string
  dataset_id: string
  build_id: string
  max_shard_tokens: number
  total_tokens: number
  total_documents: number
  excluded_document_count: number
  applied_tombstone_ids: string[]
  shard_count: number
  splits: Record<string, TokenizedSplit>
}

export interface FertilityMetrics {
  documents: number
  bytes: number
  characters: number
  words: number
  tokens: number
  byte_fallback_tokens: number
  tokens_per_byte: number | null
  tokens_per_word: number | null
  bytes_per_token: number | null
  byte_fallback_rate: number | null
}

export interface FertilityReport {
  report_id: string
  schema_version: number
  kind: string
  scope: string
  tokenizer_id: string
  tokenizer_json_sha256: string
  reserved_spec_sha256: string
  vocab_size: number
  learned_vocab_size: number
  dataset_id: string
  build_id: string
  split: string
  split_sha256: string
  in_sample: boolean
  excluded_document_count: number
  applied_tombstone_ids: string[]
  round_trip_failures: number
  overall: FertilityMetrics
  by_domain: Record<string, FertilityMetrics>
  by_language: Record<string, FertilityMetrics>
}

export interface TokenizerSummary {
  schema_version: number
  tokenizers: TokenizerArtifact[]
  tokenized_builds: TokenizedBuild[]
  fertility_reports: FertilityReport[]
  integrity_errors: string[]
}

export interface KernelEvaluation {
  split: string
  main_loss: number
  perplexity: number
  mtp_losses: number[]
  target_tokens: number
  steps: number
}

export interface KernelCheckpoint {
  name: string
  step: number
  world_size: number | null
  checkpoint_sha256: string
}

export interface KernelRun {
  run_id: string
  lineage_id: string | null
  parent_run_id: string | null
  source: 'job' | 'cli'
  location: string
  status: string
  termination_reason: string | null
  started_at: string | null
  finished_at: string | null
  model_id: string | null
  config_sha256: string | null
  qk_norm: boolean
  mtp_depth: number
  precision: string | null
  strategy: string | null
  world_size: number | null
  tokenized_id: string | null
  seq_len: number | null
  tokens_per_batch: number | null
  start_step: number | null
  target_steps: number | null
  step: number | null
  training_loss: number | null
  main_loss: number | null
  mtp_losses: number[]
  learning_rate: number | null
  tokens_seen: number | null
  tokens_per_second: number | null
  evaluation: KernelEvaluation | null
  checkpoint: KernelCheckpoint | null
  manifest_sha256: string
  objective?: 'pretrain' | 'sft' | 'logit-distillation' | 'dpo'
  chat_dataset_id?: string | null
  parent_checkpoint_sha256?: string | null
  reference_checkpoint_sha256?: string | null
  distill_teachers?: number
  distill_kl?: number | null
  reward_margin?: number | null
  preference_accuracy?: number | null
  policy_logratio?: number | null
  alerts?: string[]
  preference_evaluation?: Record<string, number | string> | null
}

export interface KernelEvaluationReport {
  location: string
  report_sha256: string
  checkpoint_step: number | null
  model_id: string | null
  tokenized_id: string | null
  split: string | null
  main_loss: number | null
  perplexity: number | null
  mtp_losses: number[]
  target_tokens: number | null
  steps: number | null
  created_at: string | null
}

export interface TrainingSummary {
  schema_version: number
  scope: string
  runs: KernelRun[]
  evaluations: KernelEvaluationReport[]
  integrity_errors: string[]
}

export interface CertifiedProfile {
  profile_id: string
  hardware: string
  max_context: number
  precision: string
  status: 'planned' | 'validated' | 'certified'
  ram_gib: number | null
  vram_gib: number | null
  notes: string | null
}

export interface ModelFamily {
  display_name: string
  slug: string
  family: 'alpha' | 'beta' | 'gamma'
  architecture: string
  total_parameters: number | null
  active_parameters: number | null
  core_parameters: number | null
  configured_max_context: number
  trained_max_context: number
  validated_max_context: number
  operational_default_context: number
  certified_profiles: CertifiedProfile[]
  research_features: string[]
  status: 'planned' | 'proxy' | 'trained'
  metrics: null
}

export interface DocumentSummary {
  slug: string
  title: string
  group: string
}

export interface DocumentRead extends DocumentSummary {
  markdown: string
}

export type Role = 'owner' | 'delegated_operator' | 'anonymous'
export type EscalationState = 'open' | 'approved' | 'refused' | 'consumed' | 'expired'

export interface Identity {
  role: Role
  principal_id: string | null
  label: string
  scopes: string[]
  authenticated: boolean
  expires_at: string | null
}

export interface Operator {
  id: string
  role: Role
  label: string
  scopes: string[]
  active: boolean
  created_at: string
  expires_at: string | null
  revoked_at: string | null
  rotated_at: string | null
}

/** The only shape that ever carries a credential value, and only once. */
export interface IssuedCredential {
  operator: Operator
  token: string
  shown_once: true
}

export interface Obligation {
  key: string
  statement: string
}

export interface Principle {
  priority: number
  key: string
  title: string
  origin: string
  law: string
  obligations: Obligation[]
}

export interface Ratification {
  present: boolean
  matches_covenant: boolean
  ratified_on: string | null
  reviewer_role: string | null
  roadmap_version: string | null
  rationale: string | null
  evidence: string[]
}

export interface EnforcementCheckpoint {
  name: string
  description: string
}

export interface Covenant {
  covenant_id: string
  covenant_version: number
  roadmap_version: string
  effective_date: string
  status: 'draft' | 'ratified'
  sha256: string
  escalation_confidence_floor: number
  principles: Principle[]
  limitations: string
  ratification: Ratification
  enforcement_checkpoints: EnforcementCheckpoint[]
}

export interface Assessment {
  priority: number
  status: string
  confidence: number
  rationale: string
  obligation: string | null
}

export interface DeclaredAction {
  action_id: string
  summary: string
  outcome: 'allow' | 'refuse' | 'escalate'
  job_type: boolean
  cli_commands: string[]
  assessments: Assessment[]
}

export interface ActionRegistry {
  items: DeclaredAction[]
  count: number
  undeclared_behavior: string
}

export interface Escalation {
  id: string
  actor_id: string
  actor_role: Role
  action_id: string
  request_digest: string
  covenant_sha256: string
  binding_priority: number | null
  reasons: string[]
  state: EscalationState
  decided_by: string | null
  rationale: string | null
  created_at: string
  decided_at: string | null
  approval_expires_at: string | null
  consumed_at: string | null
  consumed_job_id: string | null
}

export interface Job {
  id: string
  type: JobType
  status: JobStatus
  spec: Record<string, unknown>
  result: Record<string, unknown> | null
  progress: number
  error_code: string | null
  error_message: string | null
  worker_id: string | null
  retry_of_id: string | null
  submitted_by_id: string | null
  submitted_by_role: string | null
  admission_covenant_sha256: string | null
  escalation_id: string | null
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  heartbeat_at: string | null
  cancel_requested_at: string | null
}

export interface JobEvent {
  sequence: number
  kind: string
  payload: Record<string, unknown>
  created_at: string
}

export interface Artifact {
  id: string
  job_id: string
  kind: string
  path: string
  sha256: string
  size_bytes: number
  media_type: string | null
  created_at: string
}

export interface JsonSchemaProperty {
  $ref?: string
  type?: string | string[]
  title?: string
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  enum?: unknown[]
  items?: JsonSchemaProperty
  anyOf?: JsonSchemaProperty[]
}

export interface JobTypeDefinition {
  type: JobType
  title: string
  description: string
  schema: {
    properties?: Record<string, JsonSchemaProperty>
    required?: string[]
  }
}

export interface ErrorEnvelope {
  code: string
  message: string
  details: unknown
  request_id: string
}

export type MeasurementScope = 'harness-fixture' | 'proxy-scale'

export interface PrimaryMetric {
  task_id: string
  task_type: string
  metric: string
  value: number | null
  scope: string
}

export interface EvaluationReportSummary {
  report_id: string
  suite_id: string
  suite_version: number
  scope: string
  claim_boundary: string
  model_kind: string
  model_id: string
  checkpoint_step: number | null
  parameter_count: number | null
  task_count: number
  skipped_task_count: number
  primary_metrics: PrimaryMetric[]
  code_executed: boolean
}

export interface EvaluationReportList {
  schema_version: number
  reports: EvaluationReportSummary[]
  integrity_errors: string[]
}

export interface EvaluationMetric {
  name: string
  value: number | null
  scope: string
  higher_is_better: boolean
}

export interface ReliabilityBin {
  lower: number
  upper: number
  count: number
  mean_confidence: number | null
  accuracy: number | null
}

export interface CalibrationSummary {
  method: string
  probability_source: string
  bins: number
  items: number
  expected_calibration_error: number
  brier_score: number
  reliability: ReliabilityBin[]
}

export interface SandboxIsolation {
  sandbox_version: string
  network_isolated: boolean
  network_method: string
  filesystem_writes_confined: boolean
  landlock_abi: number
  signals_scoped: boolean
  process_limit_scope: string
  notes: string[]
}

export interface EvaluationTask {
  task_id: string
  task_type: string
  format: string
  primary_metric: string
  metrics: EvaluationMetric[]
  item_count: number
  settings: Record<string, unknown>
  calibration: CalibrationSummary | null
  sandbox: SandboxIsolation | null
}

export interface EvaluationReportView {
  report_id: string
  suite_id: string
  suite_version: number
  scope: string
  claim_boundary: string
  model: {
    kind: string
    model_id: string
    checkpoint_step: number | null
    parameter_count: number | null
    tokenizer_id: string
  }
  data: { kind: string; source_id: string; content_sha256: string; task_ids: string[] }[]
  tasks: EvaluationTask[]
  skipped_tasks: { task_id: string; reason: string }[]
  summary: EvaluationReportSummary
}

export interface MetricDelta {
  task_id: string
  name: string
  baseline: number | null
  candidate: number | null
  delta: number | null
  higher_is_better: boolean
}

export interface PairedBootstrap {
  task_id: string
  paired_items: number
  unmatched_items: number
  baseline_mean: number | null
  candidate_mean: number | null
  mean_delta: number | null
  ci_lower: number | null
  ci_upper: number | null
  confidence: number
  resamples: number
  seed: number
}

export interface ReportComparison {
  baseline_report_id: string
  candidate_report_id: string
  warnings: string[]
  metric_deltas: MetricDelta[]
  bootstraps: PairedBootstrap[]
}

export interface ScalingReportSummary {
  report_id: string
  content_sha256: string
  kind: string
  scope: string
  claim_boundary: string
  warning_count: number
  recorded_at: string | null
  [headline: string]: unknown
}

export interface ScalingReportList {
  schema_version: number
  reports: ScalingReportSummary[]
  integrity_errors: string[]
}

export interface LadderRung {
  slug: string
  config_sha256: string
  nominal_non_embedding: number | null
  total_parameters: number
  non_embedding_parameters: number
  vocab_size: number
  execution: 'local' | 'projection-only'
  reason: string
}

export interface Ladder {
  rungs: LadderRung[]
  note: string
}

export interface ScalingReport {
  report_id: string
  kind: string
  scope: string
  claim_boundary: string
  inputs: Record<string, unknown>
  results: Record<string, unknown>
  warnings: string[]
  measured: Record<string, unknown> | null
  summary: ScalingReportSummary
}

export type MonitorStatus = 'ok' | 'unknown' | 'alert' | 'abort'
export type G10Gate = 'blocked' | 'escalated' | 'no-regression-observed'

export interface AuditVerdictRecord {
  verdict: 'accept' | 'reject'
  actor_id: string
  actor_role: string
  note: string
  created_at: string
}

export interface AuditItem {
  id: string
  subject: 'preference' | 'revision' | 'eval-item' | 'g10-item'
  judging_run: string
  record: string
  stratum: string
  judge_accepted: boolean | null
  confidence: number | null
  created_at: string
  created_by: string
  detail: Record<string, unknown>
  state: 'pending' | 'audited'
  verdict: 'accept' | 'reject' | null
  verdicts: AuditVerdictRecord[]
}

export interface JudgeReliabilityBin {
  confidence_low: number
  confidence_high: number
  items: number
  human_agreement: number | null
}

export interface JudgingRun {
  judging_run: string
  subject: string
  judge: string
  decisions: number
  accepted: number
  rejected: number
  sampled: number
  audited: number
  pending: number
  disagreement_rate: number | null
  strata: Record<string, { sampled: number; audited: number; disagreement_rate: number | null }>
  agreement_matrix: Record<string, number>
  reliability: JudgeReliabilityBin[]
  admitted: boolean
  admission: string
  max_disagreement: number
}

export interface MonitorValue {
  value: number | null
  status: MonitorStatus
  threshold: { alert: number; abort: number }
  [key: string]: unknown
}

export interface OveroptimizationView {
  report_id: string
  status: MonitorStatus
  created_at: string
  policy: { model: string; version: string }
  reference: { model: string; version: string }
  monitors: Record<'kl_per_token' | 'proxy_minus_audited' | 'length_ratio', MonitorValue>
  evidence: string
  claim: string
}

export interface G10View {
  report_id: string
  gate: G10Gate
  created_at: string
  before: { model: string; version: string }
  after: { model: string; version: string }
  categories: Record<string, { before: number | null; after: number | null }>
  overall: { before: number; after: number }
  regressed_categories: string[]
  failing_items: string[]
  claim: string
}

export interface AlignmentDataset {
  dataset_id: string
  kind: 'sft' | 'preference'
  examples: Record<string, number>
  target_positions: Record<string, number>
  excluded: Record<string, number>
  sources: { generator: string; purpose: string; records: number }[]
  candidates_admitted: number
}

export interface AlignmentSummary {
  schema_version: number
  scope: string
  records: Record<string, number>
  examples_by_origin: Record<string, number>
  candidates: number
  datasets: AlignmentDataset[]
  judging_runs: JudgingRun[]
  audit_queue: { pending: number; audited: number }
  overoptimization: OveroptimizationView[]
  g10: G10View[]
  integrity_errors: string[]
}
