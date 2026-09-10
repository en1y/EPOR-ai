export type JobType =
  | 'system_probe'
  | 'research_sync'
  | 'data_ingest'
  | 'data_build'
  | 'data_remove'
  | 'data_audit'
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
