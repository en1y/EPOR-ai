import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import type {
  Artifact,
  Capabilities,
  Health,
  Job,
  JobEvent,
  JobStatus,
  JobType,
  JobTypeDefinition,
  JsonSchemaProperty,
  ModelFamily,
  ResearchCatalog,
} from './types'

type Page = 'overview' | 'research' | 'models' | 'jobs'

const pageCopy: Record<Page, { title: string; eyebrow: string }> = {
  overview: { title: 'System overview', eyebrow: 'Local environment' },
  research: { title: 'Research archive', eyebrow: 'Evidence ledger' },
  models: { title: 'Model families', eyebrow: 'Architecture plans' },
  jobs: { title: 'Jobs & artifacts', eyebrow: 'Local orchestration' },
}

const terminalStatuses = new Set<JobStatus>([
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
])

type FormValue = string | boolean

const focusableSelector = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

function focusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(focusableSelector)).filter(
    (element) => element.getAttribute('aria-hidden') !== 'true',
  )
}

function useDialogBehavior<T extends HTMLElement>(onClose: () => void) {
  const dialogRef = useRef<T>(null)
  const restoreFocusRef = useRef<HTMLElement | null>(
    typeof document !== 'undefined' && document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null,
  )

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    const initialFocus =
      dialog.querySelector<HTMLElement>('[data-dialog-initial-focus]') ??
      focusableElements(dialog)[0] ??
      dialog
    initialFocus.focus()

    return () => {
      const restoreFocus = restoreFocusRef.current
      if (restoreFocus?.isConnected) restoreFocus.focus()
    }
  }, [])

  const onDialogKeyDown = useCallback(
    (event: React.KeyboardEvent<T>) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        onClose()
        return
      }
      if (event.key !== 'Tab') return

      const dialog = dialogRef.current
      if (!dialog) return
      const focusable = focusableElements(dialog)
      if (focusable.length === 0) {
        event.preventDefault()
        dialog.focus()
        return
      }

      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const active = document.activeElement
      if (!dialog.contains(active)) {
        event.preventDefault()
        ;(event.shiftKey ? last : first).focus()
      } else if (event.shiftKey && (active === first || active === dialog)) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && active === last) {
        event.preventDefault()
        first.focus()
      }
    },
    [onClose],
  )

  return { dialogRef, onDialogKeyDown }
}

export function schemaPropertyType(property: JsonSchemaProperty): string {
  const directTypes = Array.isArray(property.type)
    ? property.type
    : property.type
      ? [property.type]
      : []
  const direct = directTypes.find((type) => type !== 'null')
  if (direct) return direct

  for (const alternative of property.anyOf ?? []) {
    const resolved = schemaPropertyType(alternative)
    if (resolved !== 'null') return resolved
  }
  return directTypes[0] ?? 'string'
}

function effectiveProperty(property: JsonSchemaProperty): JsonSchemaProperty {
  const type = schemaPropertyType(property)
  const alternative = property.anyOf?.find(
    (candidate) => schemaPropertyType(candidate) === type,
  )
  return { ...alternative, ...property, type }
}

export function hydrateFormDefaults(
  definition: JobTypeDefinition | undefined,
): Record<string, FormValue> {
  const defaults: Record<string, FormValue> = {}
  for (const [name, property] of Object.entries(definition?.schema.properties ?? {})) {
    if (property.default === null || property.default === undefined) {
      defaults[name] = ''
    } else if (typeof property.default === 'boolean') {
      defaults[name] = property.default
    } else if (Array.isArray(property.default)) {
      defaults[name] = property.default.join(', ')
    } else {
      defaults[name] = String(property.default)
    }
  }
  return defaults
}

export function serializeJobSpec(
  definition: JobTypeDefinition,
  values: Record<string, FormValue>,
): Record<string, unknown> {
  const spec: Record<string, unknown> = {}
  for (const [name, property] of Object.entries(definition.schema.properties ?? {})) {
    const value = values[name]
    if (value === '' || value === undefined) continue

    const type = schemaPropertyType(property)
    if (type === 'integer' || type === 'number') {
      const number = Number(value)
      if (Number.isFinite(number)) spec[name] = number
    } else if (type === 'boolean') {
      spec[name] = typeof value === 'boolean' ? value : value === 'true'
    } else if (type === 'array') {
      const entries = String(value)
        .split(',')
        .map((part) => part.trim())
        .filter(Boolean)
      if (entries.length > 0) spec[name] = entries
    } else {
      spec[name] = value
    }
  }
  return spec
}

function formatNumber(value: number | null): string {
  if (value === null) return '—'
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`
  return value.toLocaleString()
}

function formatBytes(value: number | null | unknown): string {
  if (typeof value !== 'number') return '—'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let amount = value
  let index = 0
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024
    index += 1
  }
  return `${amount.toFixed(index > 1 ? 1 : 0)} ${units[index]}`
}

function formatDate(value: string | null): string {
  if (!value) return '—'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function shortId(value: string): string {
  return value.slice(0, 8)
}

function titleCase(value: string): string {
  return value.replaceAll('_', ' ').replaceAll('-', ' ')
}

function StatusPill({ status }: { status: string }) {
  const tone = ['succeeded', 'verified', 'ok', 'certified'].includes(status)
    ? 'positive'
    : ['failed', 'error', 'interrupted'].includes(status)
      ? 'negative'
      : ['running', 'starting', 'cancelling'].includes(status)
        ? 'active'
        : 'neutral'
  return <span className={`status-pill ${tone}`}>{titleCase(status)}</span>
}

function Icon({ name }: { name: Page }) {
  if (name === 'overview') {
    return <path d="M4 5h16v12H4zM8 21h8M12 17v4M8 9h8M8 13h5" />
  }
  if (name === 'research') {
    return <path d="M5 4h11a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3V4Zm3 0v13a3 3 0 0 0-3 3M11 8h5M11 12h5" />
  }
  if (name === 'models') {
    return <path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Zm-8 9 8 4.5 8-4.5M4 16.5l8 4.5 8-4.5" />
  }
  return <path d="M5 5h14v14H5zM9 9h6M9 13h6M9 17h3" />
}

function Navigation({ page, onPage }: { page: Page; onPage: (page: Page) => void }) {
  const pages: Page[] = ['overview', 'research', 'models', 'jobs']
  return (
    <aside className="sidebar">
      <div className="brand" aria-label="EPOR AI">
        <span className="brand-mark">E</span>
        <span>
          <strong>EPOR</strong>
          <small>Research console</small>
        </span>
      </div>
      <nav aria-label="Primary navigation">
        <span className="nav-label">Workspace</span>
        {pages.map((item) => (
          <button
            type="button"
            key={item}
            className={page === item ? 'nav-item selected' : 'nav-item'}
            aria-current={page === item ? 'page' : undefined}
            onClick={() => onPage(item)}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <Icon name={item} />
            </svg>
            {pageCopy[item].title.replace('System ', '').replace(' & artifacts', '')}
          </button>
        ))}
      </nav>
      <div className="local-boundary">
        <span className="pulse-dot" />
        <span>
          <strong>Local boundary</strong>
          <small>Loopback only</small>
        </span>
      </div>
    </aside>
  )
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="empty-state">{children}</div>
}

function OverviewPage({
  health,
  capabilities,
  onQueue,
}: {
  health: Health | null
  capabilities: Capabilities | null
  onQueue: (type: JobType, spec?: Record<string, unknown>) => void
}) {
  const hardware = capabilities?.hardware ?? {}
  const cards = [
    ['API health', health?.status ?? 'unavailable', 'Control plane response'],
    ['Database', health?.database ?? 'unavailable', 'SQLite WAL index'],
    ['Python', String(hardware.python ?? '—'), 'Pinned runtime: 3.11.x'],
    ['Memory', formatBytes(hardware.memory_bytes), 'Total system RAM'],
  ]
  return (
    <div className="page-stack">
      <section className="metric-grid" aria-label="Environment summary">
        {cards.map(([label, value, note]) => (
          <article className="metric-card" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{note}</small>
          </article>
        ))}
      </section>

      <section className="split-grid">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="kicker">Environment health</span>
              <h2>Capability matrix</h2>
            </div>
            <StatusPill status={health?.status ?? 'unknown'} />
          </div>
          <dl className="fact-list">
            {[
              ['Platform', hardware.platform],
              ['Architecture', hardware.architecture],
              ['Logical CPUs', hardware.cpu_logical],
              ['uv', hardware.uv_available ? 'Available' : 'Missing'],
              ['Git', hardware.git_available ? 'Available' : 'Missing'],
              ['PyTorch', hardware.torch_version ?? 'Not installed'],
              ['CUDA', hardware.cuda_available ? 'Available' : 'Unavailable'],
              ['ROCm reported', hardware.rocm_reported ? 'Yes' : 'No'],
            ].map(([label, value]) => (
              <div key={label as string}>
                <dt>{String(label)}</dt>
                <dd>{String(value ?? '—')}</dd>
              </div>
            ))}
          </dl>
        </article>

        <article className="panel boundary-panel">
          <div className="panel-heading">
            <div>
              <span className="kicker">Security posture</span>
              <h2>Intentionally constrained</h2>
            </div>
          </div>
          <ul className="check-list">
            <li><span>✓</span>Bound to {capabilities?.bind_host ?? '127.0.0.1'}</li>
            <li><span>✓</span>Exact UI origin only</li>
            <li><span>✓</span>Four typed job definitions</li>
            <li><span>✓</span>No arbitrary shell, URL fetcher, or file browser</li>
          </ul>
          <button className="primary-button wide" type="button" onClick={() => onQueue('system_probe')}>
            Run fresh system probe
          </button>
        </article>
      </section>
    </div>
  )
}

function ResearchPage({ catalog }: { catalog: ResearchCatalog | null }) {
  const [query, setQuery] = useState('')
  const filtered = useMemo(() => {
    const value = query.trim().toLowerCase()
    if (!value) return catalog?.sources ?? []
    return (catalog?.sources ?? []).filter((source) =>
      [source.title, source.organization, source.id, ...source.tags]
        .join(' ')
        .toLowerCase()
        .includes(value),
    )
  }, [catalog, query])
  const verified = catalog?.sources.filter((item) => item.sync_status === 'verified').length ?? 0
  return (
    <section className="panel table-panel">
      <div className="panel-heading research-heading">
        <div>
          <span className="kicker">Tracked source ledger</span>
          <h2>{catalog?.count ?? 0} canonical sources</h2>
          <p>{verified} locally verified · raw and extracted content remains ignored</p>
        </div>
        <label className="search-box">
          <span className="sr-only">Filter research sources</span>
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6" /><path d="m16 16 4 4" /></svg>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter evidence…" />
        </label>
      </div>
      {!catalog?.available ? (
        <EmptyState>{catalog?.error ?? 'Research catalog unavailable.'}</EmptyState>
      ) : (
        <div className="table-scroll">
          <table>
            <thead><tr><th>Source</th><th>Organization</th><th>Evidence</th><th>Local status</th><th>Published</th></tr></thead>
            <tbody>
              {filtered.map((source) => (
                <tr key={source.id}>
                  <td className="source-cell">
                    <a href={source.canonical_url} target="_blank" rel="noreferrer">{source.title}</a>
                    <span>{source.tags.slice(0, 3).join(' · ')}</span>
                  </td>
                  <td>{source.organization}</td>
                  <td><span className="evidence-label">{source.evidence_classes[0]}</span></td>
                  <td><StatusPill status={source.sync_status} /></td>
                  <td>{source.publication_date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function ContextRow({ label, value, ceiling }: { label: string; value: number; ceiling: number }) {
  const width = value === 0 ? 0 : Math.max(4, (value / ceiling) * 100)
  return (
    <div className="context-row">
      <span>{label}</span><strong>{formatNumber(value)}</strong>
      <div className="context-track"><i style={{ width: `${width}%` }} /></div>
    </div>
  )
}

function ModelsPage({ models }: { models: ModelFamily[] }) {
  if (models.length === 0) return <EmptyState>No family configuration files were found.</EmptyState>
  return (
    <div className="model-grid">
      {models.map((model) => (
        <article className={`model-card model-${model.family}`} key={model.slug}>
          <div className="model-card-head">
            <div>
              <span className="model-symbol">{model.display_name.slice(-1)}</span>
              <span className="kicker">{titleCase(model.architecture)}</span>
              <h2>{model.display_name}</h2>
            </div>
            <StatusPill status={model.status} />
          </div>
          <p className="plan-warning">Configuration target · no trained weights or benchmark metrics</p>
          <div className="parameter-strip">
            <div><span>Total</span><strong>{formatNumber(model.total_parameters)}</strong></div>
            <div><span>Active</span><strong>{formatNumber(model.active_parameters)}</strong></div>
            <div><span>Core</span><strong>{formatNumber(model.core_parameters)}</strong></div>
          </div>
          <div className="context-list">
            <ContextRow label="Configured" value={model.configured_max_context} ceiling={model.configured_max_context} />
            <ContextRow label="Trained" value={model.trained_max_context} ceiling={model.configured_max_context} />
            <ContextRow label="Validated" value={model.validated_max_context} ceiling={model.configured_max_context} />
            <ContextRow label="Operational" value={model.operational_default_context} ceiling={model.configured_max_context} />
          </div>
          <div className="tag-cloud">
            {model.research_features.map((feature) => <span key={feature}>{titleCase(feature)}</span>)}
          </div>
          {model.certified_profiles.map((profile) => (
            <div className="profile-note" key={profile.profile_id}>
              <span>{profile.profile_id}</span>
              <strong>{formatNumber(profile.max_context)} · {profile.precision}</strong>
              <small>{profile.status} on reference hardware</small>
            </div>
          ))}
        </article>
      ))}
    </div>
  )
}

export function JobDetail({
  job,
  onClose,
  onChanged,
}: {
  job: Job
  onClose: () => void
  onChanged: () => void
}) {
  const [events, setEvents] = useState<JobEvent[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [actionError, setActionError] = useState('')
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLElement>(onClose)

  useEffect(() => {
    let disposed = false
    let cursor = 0
    setEvents([])
    const refresh = async () => {
      try {
        const response = await api.events(job.id, cursor)
        if (!disposed && response.items.length) {
          cursor = response.next_after
          setEvents((current) => {
            const known = new Set(current.map((item) => item.sequence))
            return [...current, ...response.items.filter((item) => !known.has(item.sequence))]
          })
        }
        const nextArtifacts = await api.artifacts(job.id)
        if (!disposed) setArtifacts(nextArtifacts)
      } catch (error) {
        if (!disposed) setActionError(error instanceof Error ? error.message : 'Could not refresh job detail')
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 1800)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [job.id])

  const perform = async (action: 'cancel' | 'retry') => {
    setActionError('')
    try {
      await (action === 'cancel' ? api.cancel(job.id) : api.retry(job.id))
      onChanged()
      if (action === 'retry') onClose()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : `Could not ${action} job`)
    }
  }

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside
        ref={dialogRef}
        className="job-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="job-detail-title"
        tabIndex={-1}
        onKeyDown={onDialogKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-head">
          <div><span className="kicker">Job {shortId(job.id)}</span><h2 id="job-detail-title">{titleCase(job.type)}</h2></div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close detail" data-dialog-initial-focus>×</button>
        </div>
        <div className="drawer-status">
          <StatusPill status={job.status} />
          <span>{Math.round(job.progress * 100)}%</span>
          <div className="progress-track"><i style={{ width: `${job.progress * 100}%` }} /></div>
        </div>
        <dl className="detail-facts">
          <div><dt>Created</dt><dd>{formatDate(job.created_at)}</dd></div>
          <div><dt>Worker</dt><dd>{job.worker_id ?? 'Awaiting claim'}</dd></div>
          <div><dt>Parent run</dt><dd>{job.retry_of_id ? shortId(job.retry_of_id) : '—'}</dd></div>
        </dl>
        {job.error_message && <div className="error-banner"><strong>{job.error_code}</strong>{job.error_message}</div>}
        {actionError && <div className="error-banner">{actionError}</div>}
        <div className="drawer-actions">
          {!terminalStatuses.has(job.status) && <button className="danger-button" type="button" onClick={() => void perform('cancel')}>Cancel job</button>}
          {['failed', 'cancelled', 'interrupted'].includes(job.status) && <button className="primary-button" type="button" onClick={() => void perform('retry')}>Retry as new job</button>}
        </div>
        <section className="detail-section">
          <div className="section-title"><h3>Event log</h3><span>{events.length}</span></div>
          <ol className="event-list">
            {events.map((event) => (
              <li key={event.sequence}>
                <i /><div><strong>{titleCase(event.kind.replace('job.', ''))}</strong><p>{String(event.payload.message ?? event.payload.reason ?? '')}</p><time>{formatDate(event.created_at)}</time></div>
              </li>
            ))}
            {events.length === 0 && <li className="muted-line">No events yet.</li>}
          </ol>
        </section>
        <section className="detail-section">
          <div className="section-title"><h3>Artifacts</h3><span>{artifacts.length}</span></div>
          {artifacts.map((artifact) => (
            <div className="artifact-row" key={artifact.id}>
              <div><strong>{artifact.path}</strong><span>{artifact.kind} · {formatBytes(artifact.size_bytes)}</span></div>
              <code title={artifact.sha256}>{artifact.sha256.slice(0, 12)}</code>
            </div>
          ))}
          {artifacts.length === 0 && <p className="muted-line">No registered artifacts.</p>}
        </section>
      </aside>
    </div>
  )
}

export function JobForm({
  definitions,
  onClose,
  onCreated,
}: {
  definitions: JobTypeDefinition[]
  onClose: () => void
  onCreated: (job: Job) => void
}) {
  const [type, setType] = useState<JobType>(definitions[0]?.type ?? 'system_probe')
  const definition = definitions.find((item) => item.type === type)
  const [values, setValues] = useState<Record<string, FormValue>>(() => hydrateFormDefaults(definition))
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLFormElement>(onClose)

  useEffect(() => {
    setValues(hydrateFormDefaults(definition))
  }, [definition])

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!definition) return
    setSubmitting(true)
    setError('')
    const spec = serializeJobSpec(definition, values)
    try {
      onCreated(await api.createJob(type, spec))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not queue job')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <form
        ref={dialogRef}
        className="job-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="queue-job-title"
        aria-describedby="queue-job-description"
        tabIndex={-1}
        onSubmit={(event) => void submit(event)}
        onKeyDown={onDialogKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-head"><div><span className="kicker">Allowlisted execution</span><h2 id="queue-job-title">Queue a local job</h2></div><button type="button" className="icon-button" onClick={onClose} aria-label="Close">×</button></div>
        <label className="form-field"><span>Job type</span><select aria-label="Job type" aria-describedby="queue-job-description" data-dialog-initial-focus value={type} onChange={(event) => setType(event.target.value as JobType)}>{definitions.map((item) => <option value={item.type} key={item.type}>{item.title}</option>)}</select><small id="queue-job-description">{definition?.description}</small></label>
        <div className="form-grid">
          {Object.entries(definition?.schema.properties ?? {}).map(([name, property]) => {
            const resolved = effectiveProperty(property)
            const propertyType = schemaPropertyType(resolved)
            return propertyType === 'boolean' ? (
              <label className="checkbox-field" key={name}><input type="checkbox" checked={Boolean(values[name])} onChange={(event) => setValues((current) => ({ ...current, [name]: event.target.checked }))} /><span>{property.title ?? titleCase(name)}</span></label>
            ) : (
              <label className="form-field" key={name}><span>{property.title ?? titleCase(name)}</span><input required={definition?.schema.required?.includes(name)} type={propertyType === 'integer' || propertyType === 'number' ? 'number' : 'text'} min={resolved.minimum} max={resolved.maximum} value={String(values[name] ?? '')} onChange={(event) => setValues((current) => ({ ...current, [name]: event.target.value }))} /></label>
            )
          })}
        </div>
        {error && <div className="error-banner">{error}</div>}
        <div className="modal-actions"><button type="button" className="ghost-button" onClick={onClose}>Dismiss</button><button className="primary-button" disabled={submitting} type="submit">{submitting ? 'Queuing…' : 'Queue job'}</button></div>
      </form>
    </div>
  )
}

export function JobsPage({ jobs, onSelect, onNew }: { jobs: Job[]; onSelect: (job: Job) => void; onNew: () => void }) {
  const activeCount = jobs.filter((job) => !terminalStatuses.has(job.status)).length
  return (
    <section className="panel table-panel jobs-panel">
      <div className="panel-heading">
        <div><span className="kicker">Append-only local lifecycle</span><h2>{jobs.length} jobs · {activeCount} active</h2></div>
        <button className="primary-button" type="button" onClick={onNew}>+ New job</button>
      </div>
      {jobs.length === 0 ? <EmptyState>No jobs yet. Queue a typed system probe to exercise the worker.</EmptyState> : (
        <div className="table-scroll"><table><thead><tr><th>Job</th><th>Status</th><th>Progress</th><th>Created</th><th>Worker</th></tr></thead><tbody>
          {jobs.map((job) => <tr className="clickable-row" key={job.id} onClick={() => onSelect(job)}><td className="job-name"><button type="button" className="job-row-trigger" aria-label={`Open ${titleCase(job.type)} job ${shortId(job.id)} (${titleCase(job.status)})`}><strong>{titleCase(job.type)}</strong><code>{shortId(job.id)}</code></button></td><td><StatusPill status={job.status} /></td><td><div className="table-progress"><i style={{ width: `${job.progress * 100}%` }} /></div><span className="progress-copy">{Math.round(job.progress * 100)}%</span></td><td>{formatDate(job.created_at)}</td><td>{job.worker_id?.split(':')[0] ?? '—'}</td></tr>)}
        </tbody></table></div>
      )}
    </section>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>('overview')
  const [health, setHealth] = useState<Health | null>(null)
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [research, setResearch] = useState<ResearchCatalog | null>(null)
  const [models, setModels] = useState<ModelFamily[]>([])
  const [definitions, setDefinitions] = useState<JobTypeDefinition[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [showJobForm, setShowJobForm] = useState(false)
  const [error, setError] = useState('')

  const refreshJobs = useCallback(async () => {
    try { setJobs(await api.jobs()) } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load jobs') }
  }, [])

  useEffect(() => {
    let disposed = false
    Promise.all([api.health(), api.capabilities(), api.research(), api.models(), api.jobTypes()])
      .then(([nextHealth, nextCapabilities, nextResearch, nextModels, nextDefinitions]) => {
        if (disposed) return
        setHealth(nextHealth); setCapabilities(nextCapabilities); setResearch(nextResearch); setModels(nextModels); setDefinitions(nextDefinitions)
      })
      .catch((caught: unknown) => { if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not connect to the control API') })
    return () => { disposed = true }
  }, [])

  useEffect(() => {
    void refreshJobs()
    const timer = window.setInterval(() => void refreshJobs(), 2000)
    return () => window.clearInterval(timer)
  }, [refreshJobs])

  const selectedJob = jobs.find((job) => job.id === selectedId) ?? null
  const queueQuickJob = async (type: JobType, spec: Record<string, unknown> = {}) => {
    try { const job = await api.createJob(type, spec); setPage('jobs'); setSelectedId(job.id); await refreshJobs() }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not queue job') }
  }

  return (
    <div className="app-shell">
      <Navigation page={page} onPage={setPage} />
      <main>
        <header className="topbar">
          <div><span className="eyebrow">{pageCopy[page].eyebrow}</span><h1>{pageCopy[page].title}</h1></div>
          <div className="topbar-state"><span className={health ? 'pulse-dot' : 'pulse-dot offline'} /><div><strong>{health ? `EPOR ${health.version}` : 'API unavailable'}</strong><small>{capabilities?.bind_host ?? '127.0.0.1'} · private</small></div></div>
        </header>
        {error && <div className="global-error" role="alert"><span>{error}</span><button type="button" onClick={() => setError('')}>Dismiss</button></div>}
        <div className="page-content">
          {page === 'overview' && <OverviewPage health={health} capabilities={capabilities} onQueue={(type, spec) => void queueQuickJob(type, spec)} />}
          {page === 'research' && <ResearchPage catalog={research} />}
          {page === 'models' && <ModelsPage models={models} />}
          {page === 'jobs' && <JobsPage jobs={jobs} onSelect={(job) => setSelectedId(job.id)} onNew={() => setShowJobForm(true)} />}
        </div>
      </main>
      {selectedJob && <JobDetail job={selectedJob} onClose={() => setSelectedId(null)} onChanged={() => void refreshJobs()} />}
      {showJobForm && <JobForm definitions={definitions} onClose={() => setShowJobForm(false)} onCreated={(job) => { setShowJobForm(false); setSelectedId(job.id); void refreshJobs() }} />}
    </div>
  )
}
