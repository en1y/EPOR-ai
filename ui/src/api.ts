import type {
  ActionRegistry,
  Artifact,
  Capabilities,
  Covenant,
  DocumentRead,
  DocumentSummary,
  ErrorEnvelope,
  Escalation,
  EscalationState,
  Health,
  Identity,
  IssuedCredential,
  Job,
  JobEvent,
  JobType,
  JobTypeDefinition,
  ModelFamily,
  Operator,
  ResearchCatalog,
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
