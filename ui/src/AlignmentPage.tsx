import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { EmptyState, StatusPill, formatDate, shortId, titleCase } from './shared'
import type {
  AlignmentSummary,
  AuditItem,
  G10View,
  JobType,
  JudgingRun,
  MonitorStatus,
  OveroptimizationView,
} from './types'

type StartJob = (type: JobType, values?: Record<string, string | boolean>) => void

const pollMilliseconds = 5_000

function rate(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

function monitorTone(status: MonitorStatus): string {
  return { ok: 'succeeded', unknown: 'queued', alert: 'cancelling', abort: 'failed' }[status]
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : value === undefined || value === null ? '' : JSON.stringify(value)
}

function g10Evidence(after: unknown): string {
  if (typeof after !== 'object' || after === null) return '—'
  const result = after as Record<string, unknown>
  if (typeof result.reply === 'string') return `greedy reply: ${result.reply || '(empty)'}`
  if (typeof result.margin_per_byte === 'number') {
    return `prefers the unsafe reply by ${(-result.margin_per_byte).toFixed(3)} nats per byte`
  }
  return '—'
}

/** The material a reviewer needs, shown as inert text. Model output is data, never an instruction. */
function AuditDetail({ item }: { item: AuditItem }) {
  const detail = item.detail
  const prompt = Array.isArray(detail.prompt) ? (detail.prompt as { role: string; content: string }[]) : []
  const replies = Array.isArray(detail.replies) ? (detail.replies as string[]) : []
  const preferred = typeof detail.judge_preferred === 'number' ? detail.judge_preferred : null
  return (
    <div className="audit-detail">
      <p className="audit-question">{text(detail.question)}</p>
      {prompt.length > 0 && (
        <dl className="audit-prompt">
          {prompt.map((message, index) => (
            <div key={index}>
              <dt>{message.role}</dt>
              <dd>{message.content}</dd>
            </div>
          ))}
        </dl>
      )}
      {replies.length > 0 && (
        <ol className="audit-replies">
          {replies.map((reply, index) => (
            <li key={index}>
              <span className="audit-reply-label">
                Reply {index + 1}
                {preferred === index ? ' · judge preferred' : ''}
              </span>
              <blockquote>{reply || '(empty reply)'}</blockquote>
            </li>
          ))}
        </ol>
      )}
      {item.subject === 'eval-item' && (
        <dl className="audit-prompt">
          {(['category', 'obligation', 'context', 'safe', 'unsafe'] as const).map((key) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{text(detail[key])}</dd>
            </div>
          ))}
        </dl>
      )}
      {item.subject === 'g10-item' && (
        <dl className="audit-prompt">
          <div><dt>category</dt><dd>{text(detail.category)}</dd></div>
          <div><dt>obligation</dt><dd>{text(detail.obligation)}</dd></div>
          <div><dt>situation</dt><dd>{text(detail.situation)}</dd></div>
          <div><dt>after alignment</dt><dd>{g10Evidence(detail.after)}</dd></div>
        </dl>
      )}
    </div>
  )
}

function AuditQueue({
  items,
  canAudit,
  onSignIn,
  onRecorded,
}: {
  items: AuditItem[]
  canAudit: boolean
  onSignIn: () => void
  onRecorded: () => void
}) {
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pending = items.filter((item) => item.state === 'pending')
  const audited = items.filter((item) => item.state === 'audited')

  const record = (item: AuditItem, verdict: 'accept' | 'reject') => {
    if (!canAudit) {
      onSignIn()
      return
    }
    setBusy(item.id)
    setError(null)
    api
      .auditVerdict(item.id, verdict, notes[item.id] ?? '')
      .then(() => onRecorded())
      .catch((caught: unknown) => setError(caught instanceof Error ? caught.message : 'Could not record the verdict'))
      .finally(() => setBusy(null))
  }

  return (
    <section className="panel" aria-labelledby="audit-queue-heading">
      <div className="panel-heading">
        <div>
          <h2 id="audit-queue-heading">Audit queue</h2>
          <p>
            Sampled accepted and rejected judge decisions, and G10 failures. An item becomes audited only
            when a verified human records a verdict here or with <code>epor align audit verdict</code>;
            nothing is ever marked audited automatically.
          </p>
        </div>
      </div>
      {error && <div className="error-banner" role="alert">{error}</div>}
      {!canAudit && pending.length > 0 && (
        <p className="plan-warning">Sign in with a credential that holds the <code>align_audit</code> scope to record verdicts.</p>
      )}
      {pending.length === 0 ? (
        <EmptyState>No item awaits a verdict. Sample a judging run with <code>epor align audit sample</code>.</EmptyState>
      ) : (
        <ul className="audit-list" aria-label="Pending audit items">
          {pending.map((item) => (
            <li key={item.id} className="audit-item">
              <div className="audit-header">
                <strong>{titleCase(item.subject)}</strong>
                <span>
                  {item.stratum} stratum
                  {item.confidence !== null && <> · judge confidence {rate(item.confidence, 2)}</>}
                </span>
                <code className="hash" title={item.judging_run}>run {shortId(item.judging_run)}</code>
              </div>
              <AuditDetail item={item} />
              <label className="form-field">
                <span>Reviewer note (optional)</span>
                <input
                  type="text"
                  maxLength={2000}
                  value={notes[item.id] ?? ''}
                  onChange={(event) => setNotes({ ...notes, [item.id]: event.target.value })}
                />
              </label>
              <div className="button-row">
                <button
                  type="button"
                  className="primary-button"
                  disabled={busy === item.id}
                  onClick={() => record(item, 'accept')}
                >
                  Accept
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  disabled={busy === item.id}
                  onClick={() => record(item, 'reject')}
                >
                  Reject
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {audited.length > 0 && (
        <details className="audit-history">
          <summary>{audited.length} audited item(s)</summary>
          <div className="table-scroll">
            <table aria-label="Audited items">
              <thead>
                <tr><th>Item</th><th>Subject</th><th>Stratum</th><th>Verdict</th><th>By</th><th>When</th></tr>
              </thead>
              <tbody>
                {audited.map((item) => {
                  const last = item.verdicts[item.verdicts.length - 1]
                  return (
                    <tr key={item.id}>
                      <td><code className="hash">{shortId(item.id)}</code></td>
                      <td>{titleCase(item.subject)}</td>
                      <td>{item.stratum}</td>
                      <td>{item.verdict}{item.verdicts.length > 1 ? ` (${item.verdicts.length} verdicts)` : ''}</td>
                      <td>{last?.actor_role ?? '—'}</td>
                      <td>{formatDate(last?.created_at ?? null)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </section>
  )
}

function Calibration({ runs }: { runs: JudgingRun[] }) {
  return (
    <section className="panel table-panel" aria-labelledby="calibration-heading">
      <div className="panel-heading">
        <div>
          <h2 id="calibration-heading">Judge calibration</h2>
          <p>
            Judge–human agreement on each run's audited sample. A run's candidates train only when both
            strata are audited and disagreement stays within its threshold (a hypothesis).
          </p>
        </div>
      </div>
      {runs.length === 0 ? (
        <EmptyState>No judging run is stored.</EmptyState>
      ) : (
        <div className="table-scroll">
          <table aria-label="Judging runs">
            <thead>
              <tr>
                <th>Run</th><th>Subject</th><th>Decisions</th><th>Accepted / rejected</th>
                <th>Audited / sampled</th><th>Disagreement</th><th>Admission</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.judging_run}>
                  <td><code className="hash" title={run.judging_run}>{shortId(run.judging_run)}</code> <small>{run.judge}</small></td>
                  <td>{titleCase(run.subject)}</td>
                  <td className="numeric-cell">{run.decisions}</td>
                  <td className="numeric-cell">{run.accepted} / {run.rejected}</td>
                  <td className="numeric-cell">{run.audited} / {run.sampled}</td>
                  <td className="numeric-cell">
                    {rate(run.disagreement_rate)} <small>max {rate(run.max_disagreement, 2)}</small>
                  </td>
                  <td>
                    <StatusPill status={run.admitted ? 'succeeded' : 'queued'} />{' '}
                    <small>{run.admitted ? 'admitted' : run.admission}</small>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function Monitors({ reports }: { reports: OveroptimizationView[] }) {
  return (
    <section className="panel table-panel" aria-labelledby="monitors-heading">
      <div className="panel-heading">
        <div>
          <h2 id="monitors-heading">Overoptimization monitors</h2>
          <p>KL to the reference, proxy reward versus human-audited agreement, and reply-length drift. Every threshold is a hypothesis.</p>
        </div>
      </div>
      {reports.length === 0 ? (
        <EmptyState>No overoptimization report is stored. Run <code>epor align monitor</code> after DPO.</EmptyState>
      ) : (
        <div className="table-scroll">
          <table aria-label="Overoptimization reports">
            <thead>
              <tr><th>Report</th><th>Status</th><th>Policy</th><th>KL / token</th><th>Proxy − audited</th><th>Length ratio</th></tr>
            </thead>
            <tbody>
              {reports.map((report) => (
                <tr key={report.report_id}>
                  <td><code className="hash" title={report.report_id}>{shortId(report.report_id)}</code><br /><small>{formatDate(report.created_at)}</small></td>
                  <td><StatusPill status={monitorTone(report.status)} /> <small>{report.status}</small></td>
                  <td>{report.policy.model} <small>{report.policy.version} vs {report.reference.version}</small></td>
                  {(['kl_per_token', 'proxy_minus_audited', 'length_ratio'] as const).map((name) => (
                    <td key={name} className="numeric-cell">
                      {rate(report.monitors[name].value)} <small>{report.monitors[name].status}</small>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function G10Panel({ reports }: { reports: G10View[] }) {
  const latest = reports[0]
  return (
    <section className="panel table-panel" aria-labelledby="g10-heading">
      <div className="panel-heading">
        <div>
          <h2 id="g10-heading">G10 before and after alignment</h2>
          <p>The six covenant situations on the checkpoint before and after one alignment step. Regressions fail closed; remaining failures go to the audit queue.</p>
        </div>
      </div>
      {!latest ? (
        <EmptyState>No G10 comparison is stored. Run <code>epor align g10</code>.</EmptyState>
      ) : (
        <>
          <div className="error-banner" role="note">
            <strong>Not evidence of safety</strong> {latest.claim}
          </div>
          <p className="report-footnote">
            Latest gate: <strong>{latest.gate}</strong> · {latest.before.version} → {latest.after.version} ·{' '}
            {latest.failing_items.length} failing item(s)
          </p>
          <div className="table-scroll">
            <table aria-label="G10 categories before and after">
              <thead><tr><th>Situation</th><th>Before</th><th>After</th></tr></thead>
              <tbody>
                {Object.entries(latest.categories).map(([category, values]) => (
                  <tr key={category} className={latest.regressed_categories.includes(category) ? 'regressed-row' : undefined}>
                    <td>{titleCase(category)}{latest.regressed_categories.includes(category) ? ' (regressed)' : ''}</td>
                    <td className="numeric-cell">{rate(values.before, 2)}</td>
                    <td className="numeric-cell">{rate(values.after, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  )
}

export function AlignmentPage({
  canAudit,
  onSignIn,
  onNew,
}: {
  canAudit: boolean
  onSignIn: () => void
  onNew: StartJob
}) {
  const [summary, setSummary] = useState<AlignmentSummary | null>(null)
  const [items, setItems] = useState<AuditItem[]>([])
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    Promise.all([api.alignment(), api.audits()])
      .then(([nextSummary, nextItems]) => {
        setSummary(nextSummary)
        setItems(nextItems)
        setError(null)
      })
      .catch((caught: unknown) => setError(caught instanceof Error ? caught.message : 'Could not load alignment state'))
  }, [])

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, pollMilliseconds)
    return () => window.clearInterval(timer)
  }, [refresh])

  return (
    <div className="page-stack">
      {error && <div className="error-banner" role="alert">{error}</div>}
      <p className="plan-warning">
        {summary?.scope ??
          'Proxy-scale alignment pipeline state; nothing here is evidence of alignment or safety.'}{' '}
        AI-judged preferences, critique/revision outputs, and model-written evaluation items are
        candidates until a human audits them.
      </p>
      {(summary?.integrity_errors.length ?? 0) > 0 && (
        <div className="error-banner" role="alert">
          <strong>Verification failed</strong> {summary?.integrity_errors.join('; ')}
        </div>
      )}
      <section className="panel" aria-labelledby="alignment-state-heading">
        <div className="panel-heading">
          <div>
            <h2 id="alignment-state-heading">Provenance records</h2>
            <p>Immutable teacher, judge, and preference records with their lineage.</p>
          </div>
          <div className="button-row">
            <button type="button" className="ghost-button" onClick={() => onNew('align_evaluate', { mode: 'g10' })}>
              Queue alignment job
            </button>
          </div>
        </div>
        <dl className="fact-list">
          {Object.entries(summary?.records ?? {}).map(([kind, count]) => (
            <div key={kind}><dt>{titleCase(kind)}</dt><dd>{count}</dd></div>
          ))}
          <div><dt>Candidates</dt><dd>{summary?.candidates ?? 0}</dd></div>
          <div><dt>Awaiting audit</dt><dd>{summary?.audit_queue.pending ?? 0}</dd></div>
          <div><dt>Datasets</dt><dd>{summary?.datasets.length ?? 0}</dd></div>
        </dl>
      </section>
      <AuditQueue items={items} canAudit={canAudit} onSignIn={onSignIn} onRecorded={refresh} />
      <Calibration runs={summary?.judging_runs ?? []} />
      <Monitors reports={summary?.overoptimization ?? []} />
      <G10Panel reports={summary?.g10 ?? []} />
    </div>
  )
}
