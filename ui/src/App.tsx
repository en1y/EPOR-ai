import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AccessPage, SignInDialog } from './AccessPage'
import { DocsPage } from './DocsPage'
import { SafetyPage } from './SafetyPage'
import { api } from './api'
import {
  EmptyState,
  StatusPill,
  formatDate,
  shortId,
  titleCase,
  useDialogBehavior,
} from './shared'
import type {
  Artifact,
  Capabilities,
  DeclaredAction,
  Health,
  Identity,
  Job,
  JobEvent,
  JobStatus,
  JobType,
  JobTypeDefinition,
  JsonSchemaProperty,
  ModelFamily,
  ResearchCatalog,
} from './types'

const anonymousIdentity: Identity = {
  role: 'anonymous',
  principal_id: null,
  label: '',
  scopes: [],
  authenticated: false,
  expires_at: null,
}

type Page = 'overview' | 'training' | 'jobs' | 'safety' | 'access' | 'models' | 'research' | 'docs'

const pageCopy: Record<Page, { title: string; summary: string }> = {
  overview: {
    title: 'Overview',
    summary: 'Local environment, capability facts, and the boundaries this console runs inside.',
  },
  training: {
    title: 'Training',
    summary: 'Reference training and evaluation runs, their loss, and the checkpoints they wrote.',
  },
  jobs: {
    title: 'Jobs',
    summary: 'Every queued and completed local job other than training and evaluation.',
  },
  safety: {
    title: 'Safety covenant',
    summary: 'The covenant being enforced right now, the actions declared under it, and the work it stopped.',
  },
  access: {
    title: 'Access',
    summary: 'Who may direct EPOR, which actions they hold, and how those credentials are managed.',
  },
  models: {
    title: 'Model families',
    summary: 'Configured architecture plans. No family has trained weights or measured metrics.',
  },
  research: {
    title: 'Research ledger',
    summary: 'Tracked sources, their evidence class, and whether the local copy verifies.',
  },
  docs: {
    title: 'Documentation',
    summary: 'Project documentation and research dossiers, rendered from the tracked Markdown.',
  },
}

const navGroups: { label: string; pages: Page[] }[] = [
  { label: 'Workspace', pages: ['overview', 'training', 'jobs'] },
  { label: 'Governance', pages: ['safety', 'access'] },
  { label: 'Reference', pages: ['models', 'research', 'docs'] },
]

const terminalStatuses = new Set<JobStatus>([
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
])

const trainingTypes = new Set<JobType>(['tiny_train', 'tiny_eval'])
const jobDetailPollMilliseconds = 1_800
const healthPollMilliseconds = 5_000

type FormValue = string | boolean

function progressPercentage(progress: number): number {
  return Math.max(0, Math.min(100, Math.round(progress * 100)))
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

function numberField(result: Record<string, unknown> | null, key: string): number | null {
  const value = result?.[key]
  return typeof value === 'number' ? value : null
}

function textField(result: Record<string, unknown> | null, key: string): string | null {
  const value = result?.[key]
  return typeof value === 'string' ? value : null
}

function formatLoss(value: number | null): string {
  return value === null ? '—' : value.toFixed(4)
}

function Icon({ name }: { name: Page }) {
  if (name === 'overview') {
    return <path d="M4 5h16v12H4zM8 21h8M12 17v4M8 9h8M8 13h5" />
  }
  if (name === 'training') {
    return <path d="M4 19h16M5 16l4-6 4 4 6-8" />
  }
  if (name === 'jobs') {
    return <path d="M5 5h14v14H5zM9 9h6M9 13h6M9 17h3" />
  }
  if (name === 'safety') {
    return <path d="M12 3l7 3v6c0 4.4-3 8-7 9-4-1-7-4.6-7-9V6l7-3ZM9 12l2 2 4-4" />
  }
  if (name === 'access') {
    return <path d="M12 3a4 4 0 1 1 0 8 4 4 0 0 1 0-8ZM4 21a8 8 0 0 1 16 0" />
  }
  if (name === 'models') {
    return <path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Zm-8 9 8 4.5 8-4.5M4 16.5l8 4.5 8-4.5" />
  }
  if (name === 'research') {
    return <path d="M5 4h11a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3V4Zm3 0v13a3 3 0 0 0-3 3M11 8h5M11 12h5" />
  }
  return <path d="M6 3h9l4 4v14H6zM14 3v5h5M9 13h7M9 17h5" />
}

function Navigation({
  page,
  onPage,
  health,
  capabilities,
  covenantStatus,
}: {
  page: Page
  onPage: (page: Page) => void
  health: Health | null
  capabilities: Capabilities | null
  covenantStatus: string
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">E</span>
        <span className="brand-text">
          <strong>EPOR</strong>
          <small>Research console</small>
        </span>
      </div>
      <nav aria-label="Primary navigation">
        {navGroups.map((group) => (
          <div className="nav-group" key={group.label}>
            <span className="nav-label">{group.label}</span>
            {group.pages.map((item) => (
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
                {pageCopy[item].title}
              </button>
            ))}
          </div>
        ))}
      </nav>
      <div className="version-card">
        <div className="version-line">
          <span className={health ? 'status-dot' : 'status-dot offline'} />
          <strong>EPOR {health ? health.version : 'offline'}</strong>
        </div>
        <dl>
          <div>
            <dt>API</dt>
            <dd>{health?.status ?? 'unavailable'}</dd>
          </div>
          <div>
            <dt>Database</dt>
            <dd>{health?.database ?? 'unavailable'}</dd>
          </div>
          <div>
            <dt>Bound to</dt>
            <dd>{capabilities?.bind_host ?? '127.0.0.1'}</dd>
          </div>
          <div>
            <dt>Covenant</dt>
            <dd>{covenantStatus}</dd>
          </div>
        </dl>
      </div>
    </aside>
  )
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
  const cards: [string, string, string][] = [
    ['Project version', health?.version ?? '—', 'Control plane and package version'],
    ['API health', health?.status ?? 'unavailable', 'Local control plane response'],
    ['Database', health?.database ?? 'unavailable', 'SQLite WAL index'],
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
              <h2>Environment</h2>
              <p>Reported by the last capability probe of this machine.</p>
            </div>
            <StatusPill status={health?.status ?? 'unknown'} />
          </div>
          <dl className="fact-list">
            {[
              ['Platform', hardware.platform],
              ['Architecture', hardware.architecture],
              ['Python', hardware.python],
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

        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Local boundary</h2>
              <p>What this console deliberately cannot do.</p>
            </div>
          </div>
          <ul className="check-list">
            <li>
              <span aria-hidden="true">✓</span>Bound to {capabilities?.bind_host ?? '127.0.0.1'}, loopback only
            </li>
            <li>
              <span aria-hidden="true">✓</span>Accepts requests from one exact UI origin
            </li>
            <li>
              <span aria-hidden="true">✓</span>Runs four typed job definitions and nothing else
            </li>
            <li>
              <span aria-hidden="true">✓</span>No shell, no URL fetcher, no filesystem browser
            </li>
          </ul>
          <div className="panel-footer">
            <button className="primary-button" type="button" onClick={() => onQueue('system_probe')}>
              Run a fresh system probe
            </button>
          </div>
        </article>
      </section>
    </div>
  )
}

function ResearchPage({ catalog }: { catalog: ResearchCatalog | null }) {
  const [query, setQuery] = useState('')
  const sources = catalog?.sources ?? []
  const filtered = useMemo(() => {
    const value = query.trim().toLowerCase()
    if (!value) return sources
    return sources.filter((source) =>
      [source.title, source.organization, source.id, ...source.tags]
        .join(' ')
        .toLowerCase()
        .includes(value),
    )
  }, [sources, query])
  const verified = sources.filter((item) => item.sync_status === 'verified').length

  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <h2>{catalog?.count ?? 0} tracked sources</h2>
          <p>
            {verified} verified against a reviewed digest. Downloaded bytes and extracted text stay
            ignored by Git; each row links to its canonical source.
          </p>
        </div>
        <label className="search-box">
          <span className="sr-only">Filter research sources</span>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="11" cy="11" r="6" />
            <path d="m16 16 4 4" />
          </svg>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter by title, lab, or tag"
          />
        </label>
      </div>
      {!catalog?.available ? (
        <EmptyState>{catalog?.error ?? 'Research catalog unavailable.'}</EmptyState>
      ) : filtered.length === 0 ? (
        <EmptyState>No source matches “{query}”.</EmptyState>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Organization</th>
                <th>Evidence</th>
                <th>Local copy</th>
                <th>Published</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((source) => (
                <tr key={source.id}>
                  <td className="source-cell">
                    <a href={source.canonical_url} target="_blank" rel="noreferrer noopener">
                      {source.title}
                    </a>
                    <span>{source.tags.slice(0, 4).join(' · ')}</span>
                  </td>
                  <td>{source.organization}</td>
                  <td>
                    <div className="evidence-labels">
                      {source.evidence_classes.map((label) => (
                        <span className="evidence-label" key={label}>
                          {label}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <StatusPill status={source.sync_status} />
                  </td>
                  <td className="numeric-cell">{source.publication_date}</td>
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
  const width = value === 0 ? 0 : Math.max(3, (value / ceiling) * 100)
  return (
    <div className="context-row">
      <span>{label}</span>
      <strong>{formatNumber(value)}</strong>
      <div className="context-track">
        <i style={{ width: `${width}%` }} />
      </div>
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
            <span className="model-symbol" aria-hidden="true">
              {model.display_name.slice(-1)}
            </span>
            <div>
              <h2>{model.display_name}</h2>
              <p>{titleCase(model.architecture)}</p>
            </div>
            <StatusPill status={model.status} />
          </div>
          <p className="plan-warning">
            Configuration target. No trained weights, no benchmark metrics.
          </p>
          <div className="parameter-strip">
            <div>
              <span>Total</span>
              <strong>{formatNumber(model.total_parameters)}</strong>
            </div>
            <div>
              <span>Active</span>
              <strong>{formatNumber(model.active_parameters)}</strong>
            </div>
            <div>
              <span>Core</span>
              <strong>{formatNumber(model.core_parameters)}</strong>
            </div>
          </div>
          <div className="context-list">
            <h3>Context length</h3>
            <ContextRow
              label="Configured"
              value={model.configured_max_context}
              ceiling={model.configured_max_context}
            />
            <ContextRow
              label="Trained"
              value={model.trained_max_context}
              ceiling={model.configured_max_context}
            />
            <ContextRow
              label="Validated"
              value={model.validated_max_context}
              ceiling={model.configured_max_context}
            />
            <ContextRow
              label="Operational default"
              value={model.operational_default_context}
              ceiling={model.configured_max_context}
            />
          </div>
          <div className="tag-cloud">
            {model.research_features.map((feature) => (
              <span key={feature}>{titleCase(feature)}</span>
            ))}
          </div>
          {model.certified_profiles.map((profile) => (
            <div className="profile-note" key={profile.profile_id}>
              <span>{profile.profile_id}</span>
              <strong>
                {formatNumber(profile.max_context)} · {profile.precision}
              </strong>
              <small>
                {profile.status} on {profile.hardware}
              </small>
            </div>
          ))}
        </article>
      ))}
    </div>
  )
}

export function LossChart({ points }: { points: { step: number; loss: number }[] }) {
  if (points.length < 2) return null
  const losses = points.map((point) => point.loss)
  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const span = max - min || 1
  const lastStep = points[points.length - 1].step || 1
  const path = points
    .map((point, index) => {
      const x = (point.step / lastStep) * 100
      const y = 100 - ((point.loss - min) / span) * 100
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')

  return (
    <figure className="loss-chart">
      <figcaption>
        <span>Training loss</span>
        <strong>
          {formatLoss(losses[0])} → {formatLoss(losses[losses.length - 1])}
        </strong>
      </figcaption>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={
        `Loss over ${points.length} reported steps, from ${formatLoss(losses[0])} to ` +
        `${formatLoss(losses[losses.length - 1])}.`
      }>
        <path d={path} vectorEffect="non-scaling-stroke" />
      </svg>
      <span className="chart-axis">
        <i>step 0</i>
        <i>step {lastStep}</i>
      </span>
    </figure>
  )
}

export function TrainingPage({
  jobs,
  onSelect,
  onTrain,
  onEvaluate,
}: {
  jobs: Job[]
  onSelect: (job: Job) => void
  onTrain: () => void
  onEvaluate: (checkpointPath: string) => void
}) {
  const runs = jobs.filter((job) => trainingTypes.has(job.type))
  const trainRuns = runs.filter((job) => job.type === 'tiny_train')
  const evaluations = runs.filter((job) => job.type === 'tiny_eval')
  const lastLoss = trainRuns.map((job) => numberField(job.result, 'final_loss')).find(
    (value) => value !== null,
  )

  return (
    <div className="page-stack">
      <section className="metric-grid" aria-label="Training summary">
        <article className="metric-card">
          <span>Training runs</span>
          <strong>{trainRuns.length}</strong>
          <small>Reference trainer, CPU only</small>
        </article>
        <article className="metric-card">
          <span>Evaluations</span>
          <strong>{evaluations.length}</strong>
          <small>Against the fixture corpus</small>
        </article>
        <article className="metric-card">
          <span>Latest final loss</span>
          <strong>{formatLoss(lastLoss ?? null)}</strong>
          <small>Most recent completed run</small>
        </article>
        <article className="metric-card">
          <span>Active</span>
          <strong>{runs.filter((job) => !terminalStatuses.has(job.status)).length}</strong>
          <small>Still queued or running</small>
        </article>
      </section>

      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Runs</h2>
            <p>
              Fixture-scale reference training, capped at 50M parameters and a 1 MiB corpus. These
              are contract tests for the pipeline, not model results.
            </p>
          </div>
          <button className="primary-button" type="button" onClick={onTrain}>
            Start a training run
          </button>
        </div>
        {runs.length === 0 ? (
          <EmptyState>
            No training runs yet. Start one to exercise the trainer end to end.
          </EmptyState>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Steps</th>
                  <th>Loss</th>
                  <th>Perplexity</th>
                  <th>Started</th>
                  <th>
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {runs.map((job) => {
                  const isTrain = job.type === 'tiny_train'
                  const checkpoint = textField(job.result, 'checkpoint_path')
                  const steps = isTrain
                    ? numberField(job.result, 'steps_completed')
                    : numberField(job.result, 'batches')
                  return (
                    <tr className="clickable-row" key={job.id} onClick={() => onSelect(job)}>
                      <td className="job-name">
                        <button
                          type="button"
                          className="job-row-trigger"
                          aria-label={`Open ${titleCase(job.type)} run ${shortId(job.id)} (${titleCase(job.status)})`}
                        >
                          <strong>{isTrain ? 'Training' : 'Evaluation'}</strong>
                          <code>{shortId(job.id)}</code>
                        </button>
                      </td>
                      <td>
                        <StatusPill status={job.status} />
                      </td>
                      <td className="numeric-cell">
                        {steps === null ? '—' : steps.toLocaleString()}
                        {!isTrain && steps !== null ? ' batches' : ''}
                      </td>
                      <td className="numeric-cell">
                        {formatLoss(numberField(job.result, isTrain ? 'final_loss' : 'loss'))}
                      </td>
                      <td className="numeric-cell">
                        {formatLoss(numberField(job.result, 'perplexity'))}
                      </td>
                      <td className="numeric-cell">{formatDate(job.created_at)}</td>
                      <td className="action-cell">
                        {isTrain && checkpoint && (
                          <button
                            type="button"
                            className="ghost-button"
                            onClick={(event) => {
                              event.stopPropagation()
                              onEvaluate(checkpoint)
                            }}
                          >
                            Evaluate
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

export function JobDetail({
  job,
  onClose,
  onChanged,
  canMutate = true,
}: {
  job: Job
  onClose: () => void
  onChanged: () => void
  canMutate?: boolean
}) {
  const [events, setEvents] = useState<JobEvent[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [actionError, setActionError] = useState('')
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLElement>(onClose)

  useEffect(() => {
    let disposed = false
    let cursor = 0
    setEvents([])
    setArtifacts([])
    const refresh = async () => {
      try {
        const response = await api.events(job.id, cursor)
        if (disposed) return
        cursor = Math.max(
          cursor,
          response.next_after,
          ...response.items.map((item) => item.sequence),
        )
        if (response.items.length) {
          setEvents((current) => {
            const merged = new Map(current.map((item) => [item.sequence, item]))
            for (const item of response.items) {
              if (!merged.has(item.sequence)) merged.set(item.sequence, item)
            }
            return [...merged.values()].sort((left, right) => left.sequence - right.sequence)
          })
        }
        const nextArtifacts = await api.artifacts(job.id)
        if (!disposed) setArtifacts(nextArtifacts)
      } catch (error) {
        if (!disposed) setActionError(error instanceof Error ? error.message : 'Could not refresh job detail')
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), jobDetailPollMilliseconds)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [job.id])

  const lossPoints = useMemo(
    () =>
      events.flatMap((event) => {
        const { step, loss } = event.payload as { step?: unknown; loss?: unknown }
        return typeof step === 'number' && typeof loss === 'number' ? [{ step, loss }] : []
      }),
    [events],
  )
  const progress = progressPercentage(job.progress)

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
          <div>
            <h2 id="job-detail-title">{titleCase(job.type)}</h2>
            <code>{shortId(job.id)}</code>
          </div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close detail" data-dialog-initial-focus>×</button>
        </div>
        <div className="drawer-status">
          <StatusPill status={job.status} />
          <span>{progress}%</span>
          <div
            className="progress-track"
            role="progressbar"
            aria-label={`${titleCase(job.type)} job progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
            aria-valuetext={`${progress}%`}
          >
            <i aria-hidden="true" style={{ width: `${progress}%` }} />
          </div>
        </div>
        <dl className="detail-facts">
          <div><dt>Created</dt><dd>{formatDate(job.created_at)}</dd></div>
          <div><dt>Worker</dt><dd>{job.worker_id ?? 'Awaiting claim'}</dd></div>
          <div><dt>Parent run</dt><dd>{job.retry_of_id ? shortId(job.retry_of_id) : '—'}</dd></div>
          <div><dt>Submitted by</dt><dd>{job.submitted_by_id ? `${titleCase(job.submitted_by_role ?? 'operator')} ${shortId(job.submitted_by_id)}` : 'Before identity existed'}</dd></div>
          <div><dt>Admitted under</dt><dd>{job.admission_covenant_sha256 ? <code className="hash">{job.admission_covenant_sha256}</code> : '—'}</dd></div>
        </dl>
        {job.error_message && <div className="error-banner"><strong>{job.error_code}</strong>{job.error_message}</div>}
        {actionError && <div className="error-banner">{actionError}</div>}
        <div className="drawer-actions">
          {!terminalStatuses.has(job.status) && <button className="danger-button" type="button" disabled={!canMutate} title={canMutate ? undefined : 'Sign in to stop this job'} onClick={() => void perform('cancel')}>Cancel job</button>}
          {['failed', 'cancelled', 'interrupted'].includes(job.status) && <button className="primary-button" type="button" disabled={!canMutate} title={canMutate ? undefined : 'Sign in to retry this job'} onClick={() => void perform('retry')}>Retry as new job</button>}
        </div>
        {lossPoints.length > 1 && (
          <section className="detail-section">
            <LossChart points={lossPoints} />
          </section>
        )}
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
  initialType,
  initialValues,
  onClose,
  onCreated,
}: {
  definitions: JobTypeDefinition[]
  initialType?: JobType
  initialValues?: Record<string, FormValue>
  onClose: () => void
  onCreated: (job: Job) => void
}) {
  const [type, setType] = useState<JobType>(initialType ?? definitions[0]?.type ?? 'system_probe')
  const definition = definitions.find((item) => item.type === type)
  // Captured once: a caller may pass a fresh object on every render.
  const seedRef = useRef(initialValues)
  const [values, setValues] = useState<Record<string, FormValue>>(() => ({
    ...hydrateFormDefaults(definition),
    ...(definition?.type === initialType ? seedRef.current : undefined),
  }))
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLFormElement>(onClose)

  useEffect(() => {
    setValues({
      ...hydrateFormDefaults(definition),
      ...(definition?.type === initialType ? seedRef.current : undefined),
    })
  }, [definition, initialType])

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
        <div className="drawer-head"><div><h2 id="queue-job-title">Queue a local job</h2><p>Only the four allowlisted job types can be started.</p></div><button type="button" className="icon-button" onClick={onClose} aria-label="Close">×</button></div>
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

export function JobsPage({ jobs, onSelect, onNew, canQueue = true }: { jobs: Job[]; onSelect: (job: Job) => void; onNew: () => void; canQueue?: boolean }) {
  const activeCount = jobs.filter((job) => !terminalStatuses.has(job.status)).length
  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <h2>{jobs.length} jobs · {activeCount} active</h2>
          <p>Append-only lifecycle. Training and evaluation runs have their own page.</p>
        </div>
        <button className="primary-button" type="button" onClick={onNew}>{canQueue ? 'New job' : 'Sign in to queue'}</button>
      </div>
      {jobs.length === 0 ? <EmptyState>No jobs yet. Queue a typed system probe to exercise the worker.</EmptyState> : (
        <div className="table-scroll"><table><thead><tr><th>Job</th><th>Status</th><th>Progress</th><th>Created</th><th>Worker</th></tr></thead><tbody>
          {jobs.map((job) => {
            const progress = progressPercentage(job.progress)
            return <tr className="clickable-row" key={job.id} onClick={() => onSelect(job)}><td className="job-name"><button type="button" className="job-row-trigger" aria-label={`Open ${titleCase(job.type)} job ${shortId(job.id)} (${titleCase(job.status)})`}><strong>{titleCase(job.type)}</strong><code>{shortId(job.id)}</code></button></td><td><StatusPill status={job.status} /></td><td><div className="table-progress" role="progressbar" aria-label={`${titleCase(job.type)} job ${shortId(job.id)} progress`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress} aria-valuetext={`${progress}%`}><i aria-hidden="true" style={{ width: `${progress}%` }} /></div><span className="progress-copy">{progress}%</span></td><td className="numeric-cell">{formatDate(job.created_at)}</td><td>{job.worker_id?.split(':')[0] ?? '—'}</td></tr>
          })}
        </tbody></table></div>
      )}
    </section>
  )
}

/**
 * Reading is anonymous throughout. A control that would change something is
 * shown, but disabled or redirected to sign-in, so the boundary is visible
 * rather than discovered by clicking and receiving a 401.
 */
export default function App() {
  const [page, setPage] = useState<Page>('overview')
  const [identity, setIdentity] = useState<Identity>(anonymousIdentity)
  const [actions, setActions] = useState<DeclaredAction[]>([])
  const [covenantStatus, setCovenantStatus] = useState('unknown')
  const [signingIn, setSigningIn] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [research, setResearch] = useState<ResearchCatalog | null>(null)
  const [models, setModels] = useState<ModelFamily[]>([])
  const [definitions, setDefinitions] = useState<JobTypeDefinition[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [jobForm, setJobForm] = useState<
    { type?: JobType; values?: Record<string, FormValue> } | null
  >(null)
  const [error, setError] = useState('')

  const refreshJobs = useCallback(async () => {
    try { setJobs(await api.jobs()) } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load jobs') }
  }, [])

  const refreshIdentity = useCallback(async () => {
    try { setIdentity(await api.identity()) } catch { setIdentity(anonymousIdentity) }
  }, [])

  useEffect(() => { void refreshIdentity() }, [refreshIdentity])

  useEffect(() => {
    let disposed = false
    api.actions()
      .then((registry) => { if (!disposed) setActions(registry.items) })
      .catch(() => undefined)
    api.covenant()
      .then((covenant) => {
        if (disposed) return
        setCovenantStatus(
          covenant.status === 'ratified' && covenant.ratification.matches_covenant
            ? `v${covenant.covenant_version} ratified`
            : `v${covenant.covenant_version} draft`,
        )
      })
      .catch(() => undefined)
    return () => { disposed = true }
  }, [])

  useEffect(() => {
    let disposed = false
    Promise.all([api.capabilities(), api.research(), api.models(), api.jobTypes()])
      .then(([nextCapabilities, nextResearch, nextModels, nextDefinitions]) => {
        if (disposed) return
        setCapabilities(nextCapabilities); setResearch(nextResearch); setModels(nextModels); setDefinitions(nextDefinitions)
      })
      .catch((caught: unknown) => { if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not connect to the control API') })
    return () => { disposed = true }
  }, [])

  useEffect(() => {
    let disposed = false
    let timer: number | undefined
    const refreshHealth = async () => {
      try {
        const nextHealth = await api.health()
        if (!disposed) setHealth(nextHealth)
      } catch {
        if (!disposed) setHealth(null)
      } finally {
        if (!disposed) {
          timer = window.setTimeout(() => void refreshHealth(), healthPollMilliseconds)
        }
      }
    }
    void refreshHealth()
    return () => {
      disposed = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    void refreshJobs()
    const timer = window.setInterval(() => void refreshJobs(), 2000)
    return () => window.clearInterval(timer)
  }, [refreshJobs])

  const selectedJob = jobs.find((job) => job.id === selectedId) ?? null
  const queueQuickJob = async (type: JobType, spec: Record<string, unknown> = {}) => {
    if (!identity.authenticated) { setSigningIn(true); return }
    try { const job = await api.createJob(type, spec); setPage('jobs'); setSelectedId(job.id); await refreshJobs() }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not queue job') }
  }
  const startJob = (next: { type?: JobType; values?: Record<string, FormValue> }) => {
    if (identity.authenticated) setJobForm(next)
    else setSigningIn(true)
  }

  return (
    <div className="app-shell">
      <Navigation
        page={page}
        onPage={setPage}
        health={health}
        capabilities={capabilities}
        covenantStatus={covenantStatus}
      />
      <main>
        <header className="topbar">
          <div>
            <h1>{pageCopy[page].title}</h1>
            <p>{pageCopy[page].summary}</p>
          </div>
          <div className="identity-control">
            <button
              type="button"
              className="identity-chip"
              onClick={() => setPage('access')}
              aria-label={`Signed in as ${identity.authenticated ? identity.label || identity.role : 'anonymous'}. Open access settings.`}
            >
              <StatusPill status={identity.role} />
              <span>{identity.authenticated ? identity.label || titleCase(identity.role) : 'Anonymous'}</span>
            </button>
            {identity.authenticated ? (
              <button
                type="button"
                className="ghost-button"
                onClick={() => void api.closeSession().finally(() => void refreshIdentity())}
              >
                Sign out
              </button>
            ) : (
              <button type="button" className="ghost-button" onClick={() => setSigningIn(true)}>
                Sign in
              </button>
            )}
          </div>
        </header>
        {error && <div className="global-error" role="alert"><span>{error}</span><button type="button" onClick={() => setError('')}>Dismiss</button></div>}
        <div className="page-content">
          {page === 'overview' && <OverviewPage health={health} capabilities={capabilities} onQueue={(type, spec) => void queueQuickJob(type, spec)} />}
          {page === 'training' && (
            <TrainingPage
              jobs={jobs}
              onSelect={(job) => setSelectedId(job.id)}
              onTrain={() => startJob({ type: 'tiny_train' })}
              onEvaluate={(checkpointPath) =>
                startJob({ type: 'tiny_eval', values: { checkpoint_path: checkpointPath } })
              }
            />
          )}
          {page === 'jobs' && <JobsPage jobs={jobs.filter((job) => !trainingTypes.has(job.type))} onSelect={(job) => setSelectedId(job.id)} onNew={() => startJob({})} canQueue={identity.authenticated} />}
          {page === 'safety' && <SafetyPage identity={identity} />}
          {page === 'access' && (
            <AccessPage
              identity={identity}
              actions={actions}
              onIdentityChanged={() => void refreshIdentity()}
              onSignIn={() => setSigningIn(true)}
            />
          )}
          {page === 'models' && <ModelsPage models={models} />}
          {page === 'research' && <ResearchPage catalog={research} />}
          {page === 'docs' && <DocsPage />}
        </div>
      </main>
      {selectedJob && <JobDetail job={selectedJob} onClose={() => setSelectedId(null)} onChanged={() => void refreshJobs()} canMutate={identity.authenticated} />}
      {signingIn && (
        <SignInDialog
          onClose={() => setSigningIn(false)}
          onSignedIn={() => { setSigningIn(false); void refreshIdentity() }}
        />
      )}
      {jobForm && (
        <JobForm
          definitions={definitions}
          initialType={jobForm.type}
          initialValues={jobForm.values}
          onClose={() => setJobForm(null)}
          onCreated={(job) => {
            setJobForm(null)
            setSelectedId(job.id)
            void refreshJobs()
          }}
        />
      )}
    </div>
  )
}
