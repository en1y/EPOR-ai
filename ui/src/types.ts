export type JobType = 'system_probe' | 'research_sync' | 'tiny_train' | 'tiny_eval'
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
  sync_status: 'verified' | 'missing' | 'error' | 'unknown'
  sha256: string | null
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
  family: 'gamma' | 'alpha' | 'beta'
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
