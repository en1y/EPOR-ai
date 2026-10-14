import type {
  ActionRegistry,
  AlignmentSummary,
  AuditItem,
  Artifact,
  Capabilities,
  Covenant,
  DataSummary,
  DocumentRead,
  DocumentSummary,
  ErrorEnvelope,
  EvaluationReportList,
  EvaluationReportView,
  Escalation,
  EscalationState,
  Health,
  Identity,
  IssuedCredential,
  Job,
  JobEvent,
  JobType,
  JobTypeDefinition,
  Ladder,
  ModelFamily,
  Operator,
  ReportComparison,
  ResearchCatalog,
  ScalingReport,
  ScalingReportList,
  TokenizerSummary,
  TrainingSummary,
} from './types'

const API = '/api/v1'

export class ApiError extends Error {
  readonly code: string
  readonly requestId: string

  constructor(payload: ErrorEnvelope, status: number) {
    super(`${payload.message} (${status})`)
    this.name = 'ApiError'
    this.code = payload.code
    this.requestId = payload.request_id
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    // The session cookie is HttpOnly and cross-origin from the dev server, so
    // it is only attached when credentials are explicitly included.
    credentials: 'include',
    headers: { Accept: 'application/json', ...init?.headers },
  })
  if (response.status === 204) return undefined as T
  const body = (await response.json()) as T | ErrorEnvelope
  if (!response.ok) throw new ApiError(body as ErrorEnvelope, response.status)
  return body as T
}

function send<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

export const api = {
  health: () => request<Health>('/health'),
  capabilities: () => request<Capabilities>('/capabilities'),
  research: () => request<ResearchCatalog>('/research'),
  data: () => request<DataSummary>('/data'),
  tokenizers: () => request<TokenizerSummary>('/tokenizers'),
  trainingRuns: () => request<TrainingSummary>('/training/runs'),
  evaluationReports: () => request<EvaluationReportList>('/evaluation/reports'),
  evaluationReport: (id: string) =>
    request<EvaluationReportView>(`/evaluation/reports/${encodeURIComponent(id)}`),
  compareReports: (baseline: string, candidate: string) =>
    request<ReportComparison>(
      `/evaluation/compare?baseline=${encodeURIComponent(baseline)}&candidate=${encodeURIComponent(candidate)}`,
    ),
  scalingReports: () => request<ScalingReportList>('/scaling/reports'),
  scalingReport: (id: string) =>
    request<ScalingReport[]>(`/scaling/reports/${encodeURIComponent(id)}`),
  ladder: () => request<Ladder>('/scaling/ladder'),
  alignment: () => request<AlignmentSummary>('/alignment'),
  audits: (state?: 'pending' | 'audited') =>
    request<AuditItem[]>(state ? `/alignment/audits?state=${state}` : '/alignment/audits'),
  auditVerdict: (id: string, verdict: 'accept' | 'reject', note: string) =>
    send<AuditItem>(`/alignment/audits/${encodeURIComponent(id)}/verdict`, { verdict, note }),
  models: async () => (await request<{ items: ModelFamily[] }>('/models')).items,
  jobTypes: async () =>
    (await request<{ items: JobTypeDefinition[] }>('/job-types')).items,
  documents: async () =>
    (await request<{ items: DocumentSummary[] }>('/documents')).items,
  document: (slug: string) =>
    request<DocumentRead>(`/documents/${slug.split('/').map(encodeURIComponent).join('/')}`),
  jobs: async () => (await request<{ items: Job[] }>('/jobs?limit=200')).items,
  job: (id: string) => request<Job>(`/jobs/${id}`),
  events: (id: string, after: number) =>
    request<{ items: JobEvent[]; next_after: number }>(
      `/jobs/${id}/events?after=${after}`,
    ),
  artifacts: async (id: string) =>
    (await request<{ items: Artifact[] }>(`/jobs/${id}/artifacts`)).items,
  createJob: (type: JobType, spec: Record<string, unknown>) =>
    send<Job>('/jobs', { type, spec }),
  cancel: (id: string) => send<Job>(`/jobs/${id}/cancel`),
  retry: (id: string) => send<Job>(`/jobs/${id}/retry`),

  identity: () => request<Identity>('/auth/me'),
  openSession: (token: string) => send<Identity>('/auth/session', { token }),
  closeSession: () => request<void>('/auth/session', { method: 'DELETE' }),
  operators: async () => (await request<{ items: Operator[] }>('/auth/operators')).items,
  createOperator: (label: string, scopes: string[], expiresAt: string) =>
    send<IssuedCredential>('/auth/operators', { label, scopes, expires_at: expiresAt }),
  rotateOperator: (id: string) => send<IssuedCredential>(`/auth/operators/${id}/rotate`),
  revokeOperator: (id: string) => send<Operator>(`/auth/operators/${id}/revoke`),
  rotateOwner: () => send<IssuedCredential>('/auth/owner/rotate'),

  covenant: () => request<Covenant>('/safety/covenant'),
  actions: () => request<ActionRegistry>('/safety/actions'),
  escalations: async (state?: EscalationState) =>
    (
      await request<{ items: Escalation[] }>(
        state ? `/safety/escalations?state=${state}` : '/safety/escalations',
      )
    ).items,
  decideEscalation: (id: string, approve: boolean, rationale: string) =>
    send<Escalation>(`/safety/escalations/${id}/decision`, { approve, rationale }),
}
