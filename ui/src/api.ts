import type {
  Artifact,
  Capabilities,
  ErrorEnvelope,
  Health,
  Job,
  JobEvent,
  JobType,
  JobTypeDefinition,
  ModelFamily,
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
    headers: { Accept: 'application/json', ...init?.headers },
  })
  const body = (await response.json()) as T | ErrorEnvelope
  if (!response.ok) throw new ApiError(body as ErrorEnvelope, response.status)
  return body as T
}

export const api = {
  health: () => request<Health>('/health'),
  capabilities: () => request<Capabilities>('/capabilities'),
  research: () => request<ResearchCatalog>('/research'),
  models: async () => (await request<{ items: ModelFamily[] }>('/models')).items,
  jobTypes: async () =>
    (await request<{ items: JobTypeDefinition[] }>('/job-types')).items,
  jobs: async () => (await request<{ items: Job[] }>('/jobs?limit=200')).items,
  job: (id: string) => request<Job>(`/jobs/${id}`),
  events: (id: string, after: number) =>
    request<{ items: JobEvent[]; next_after: number }>(
      `/jobs/${id}/events?after=${after}`,
    ),
  artifacts: async (id: string) =>
    (await request<{ items: Artifact[] }>(`/jobs/${id}/artifacts`)).items,
  createJob: (type: JobType, spec: Record<string, unknown>) =>
    request<Job>('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, spec }),
    }),
  cancel: (id: string) => request<Job>(`/jobs/${id}/cancel`, { method: 'POST' }),
  retry: (id: string) => request<Job>(`/jobs/${id}/retry`, { method: 'POST' }),
}
