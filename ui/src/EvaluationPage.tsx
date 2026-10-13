import { useEffect, useState } from 'react'
import { api } from './api'
import { EmptyState, StatusPill, shortId, titleCase } from './shared'
import type {
  EvaluationMetric,
  EvaluationReportList,
  EvaluationReportView,
  EvaluationTask,
  JobType,
  ReportComparison,
} from './types'

type StartJob = (type: JobType, values?: Record<string, string | boolean>) => void

const pollMilliseconds = 5_000

export function formatMetric(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

function metricValue(task: EvaluationTask, name: string): number | null {
  return task.metrics.find((metric) => metric.name === name)?.value ?? null
}

function MetricsTable({ metrics, label }: { metrics: EvaluationMetric[]; label: string }) {
  return (
    <div className="table-scroll">
      <table aria-label={label}>
        <thead>
          <tr>
            <th>Metric</th><th>Value</th><th>Scope</th><th>Direction</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric.name}>
              <td><code>{metric.name}</code></td>
              <td className="numeric-cell">{formatMetric(metric.value)}</td>
              <td>{metric.scope}</td>
              <td>{metric.higher_is_better ? 'higher is better' : 'lower is better'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SafetyCategories({ task }: { task: EvaluationTask }) {
  const prefix = `${task.primary_metric}.category.`
  const categories = task.metrics.filter((metric) => metric.name.startsWith(prefix))
  const note = typeof task.settings.claim_note === 'string' ? task.settings.claim_note : null
  const rubric = typeof task.settings.rubric_note === 'string' ? task.settings.rubric_note : null
  return (
    <>
      <div className="error-banner" role="note">
        <strong>Not evidence of safety</strong>
        {note ??
          'Harness for model behaviour on project-authored scenarios; a passing score on a small proxy model proves nothing about safety.'}
        {rubric && <> {rubric}</>}
      </div>
      <div className="table-scroll">
        <table aria-label={`${task.task_id} G10 categories`}>
          <thead>
            <tr>
              <th>G10 category</th><th>{task.primary_metric}</th><th>Scope</th>
            </tr>
          </thead>
          <tbody>
            {categories.map((metric) => (
              <tr key={metric.name}>
                <td>{titleCase(metric.name.slice(prefix.length))}</td>
                <td className="numeric-cell">{formatMetric(metric.value)}</td>
                <td>{metric.scope}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function Calibration({ task }: { task: EvaluationTask }) {
  const calibration = task.calibration
  if (!calibration) return null
  return (
    <>
      <p className="report-footnote">
        Calibration over {calibration.items} item(s): ECE{' '}
        {formatMetric(calibration.expected_calibration_error)} · Brier{' '}
        {formatMetric(calibration.brier_score)} · {calibration.bins} equal-width bins of{' '}
        {calibration.probability_source}
      </p>
      <div className="table-scroll">
        <table aria-label={`${task.task_id} reliability bins`}>
          <thead>
            <tr>
              <th>Confidence bin</th><th>Items</th><th>Mean confidence</th><th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {calibration.reliability.map((bin) => (
              <tr key={bin.lower}>
                <td>{bin.lower.toFixed(2)}–{bin.upper.toFixed(2)}</td>
                <td className="numeric-cell">{bin.count}</td>
                <td className="numeric-cell">{formatMetric(bin.mean_confidence, 3)}</td>
                <td className="numeric-cell">{formatMetric(bin.accuracy, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function Highlights({ task }: { task: EvaluationTask }) {
  if (task.task_type === 'contamination') {
    const flagged = metricValue(task, 'flagged_items')
    const assessed = metricValue(task, 'assessed_items')
    return (
      <p className="plan-warning report-scope">
        <strong>
          {flagged === null ? '—' : flagged.toLocaleString()} of{' '}
          {assessed === null ? '—' : assessed.toLocaleString()} assessed item(s) flagged
        </strong>{' '}
        (rate {formatMetric(metricValue(task, 'flagged_rate'), 3)}). N-gram overlap is a lower
        bound on contamination, never a clearance.
      </p>
    )
  }
  if (task.task_type === 'memorization') {
    return (
      <p className="plan-warning report-scope">
        <strong>
          Exact-suffix extraction: train {formatMetric(metricValue(task, 'train.exact_suffix_rate'), 3)}{' '}
          vs held-out {formatMetric(metricValue(task, 'heldout.exact_suffix_rate'), 3)} · gap{' '}
          {formatMetric(metricValue(task, 'extraction_gap'), 3)}
        </strong>{' '}
        A train rate above the held-out rate suggests memorization of training text; a small proxy
        model may show none.
      </p>
    )
  }
  return null
}

function TaskPanel({ task }: { task: EvaluationTask }) {
  const primary = metricValue(task, task.primary_metric)
  return (
    <article className="evaluation-task" aria-label={`Task ${task.task_id}`}>
      <div className="panel-heading">
        <div>
          <h3>{task.task_id}</h3>
          <p>
            {titleCase(task.task_type)} · {task.format} · {task.item_count} item(s)
          </p>
        </div>
        <div className="metric-inline">
          <span>{task.primary_metric}</span>
          <strong>{formatMetric(primary)}</strong>
        </div>
      </div>
      <Highlights task={task} />
      {task.task_type.startsWith('safety-') && <SafetyCategories task={task} />}
      <Calibration task={task} />
      {task.sandbox && (
        <p className="report-footnote">
          Sandbox {task.sandbox.sandbox_version}: network{' '}
          {task.sandbox.network_isolated ? `isolated (${task.sandbox.network_method})` : 'NOT isolated'} ·
          writes {task.sandbox.filesystem_writes_confined ? 'confined' : 'not confined'} · Landlock
          ABI {task.sandbox.landlock_abi} · process limit {task.sandbox.process_limit_scope}.
          Defense in depth, not a security boundary.
        </p>
      )}
      <details>
        <summary>All {task.metrics.length} metrics</summary>
        <MetricsTable metrics={task.metrics} label={`${task.task_id} metrics`} />
      </details>
    </article>
  )
}

function ReportDetail({ view }: { view: EvaluationReportView }) {
  return (
    <section className="panel" aria-label={`Evaluation report ${shortId(view.report_id)}`}>
      <div className="panel-heading">
        <div>
          <h2>
            {view.suite_id} v{view.suite_version} · {view.model.model_id}
            {view.model.checkpoint_step !== null && ` step ${view.model.checkpoint_step}`}
          </h2>
          <p>
            Report <code className="hash" title={view.report_id}>{shortId(view.report_id)}</code> ·{' '}
            {view.model.kind} · scope {view.scope}
          </p>
        </div>
        <div>
          <StatusPill status={view.scope} />
        </div>
      </div>
      <p className="plan-warning report-scope">
        <strong>Claim boundary.</strong> {view.claim_boundary}
      </p>
      {view.skipped_tasks.length > 0 && (
        <ul className="plain-list" aria-label="Skipped tasks">
          {view.skipped_tasks.map((item) => (
            <li key={item.task_id}>
              Skipped <code>{item.task_id}</code>: {item.reason}
            </li>
          ))}
        </ul>
      )}
      {view.tasks.map((task) => (
        <TaskPanel task={task} key={task.task_id} />
      ))}
    </section>
  )
}

function ComparisonPanel({ comparison }: { comparison: ReportComparison }) {
  return (
    <section className="panel" aria-label="Report comparison">
      <div className="panel-heading">
        <div>
          <h2>Comparison</h2>
          <p>
            Candidate <code className="hash">{shortId(comparison.candidate_report_id)}</code> minus
            baseline <code className="hash">{shortId(comparison.baseline_report_id)}</code>, with a
            seeded paired bootstrap over shared items.
          </p>
        </div>
      </div>
      {comparison.warnings.length > 0 && (
        <div className="error-banner" role="alert">
          <strong>Not directly comparable</strong>
          {comparison.warnings.join('; ')}
        </div>
      )}
      <div className="table-scroll">
        <table aria-label="Metric deltas">
          <thead>
            <tr>
              <th>Task</th><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th>
            </tr>
          </thead>
          <tbody>
            {comparison.metric_deltas.map((delta) => (
              <tr key={`${delta.task_id}-${delta.name}`}>
                <td>{delta.task_id}</td>
                <td><code>{delta.name}</code></td>
                <td className="numeric-cell">{formatMetric(delta.baseline)}</td>
                <td className="numeric-cell">{formatMetric(delta.candidate)}</td>
                <td className="numeric-cell">{formatMetric(delta.delta)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-scroll">
        <table aria-label="Paired bootstrap">
          <thead>
            <tr>
              <th>Task</th><th>Paired items</th><th>Mean delta</th><th>Interval</th>
            </tr>
          </thead>
          <tbody>
            {comparison.bootstraps.map((item) => (
              <tr key={item.task_id}>
                <td>{item.task_id}</td>
                <td className="numeric-cell">
                  {item.paired_items}
                  {item.unmatched_items > 0 && ` (+${item.unmatched_items} unmatched)`}
                </td>
                <td className="numeric-cell">{formatMetric(item.mean_delta)}</td>
                <td className="numeric-cell">
                  {formatMetric(item.ci_lower)} to {formatMetric(item.ci_upper)} at{' '}
                  {Math.round(item.confidence * 100)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export function EvaluationPage({
  onNew,
  checkpointPath,
  tokenizedId,
}: {
  onNew: StartJob
  checkpointPath: string | null
  tokenizedId: string | null
}) {
  const [listing, setListing] = useState<EvaluationReportList | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<EvaluationReportView | null>(null)
  const [baseline, setBaseline] = useState('')
  const [candidate, setCandidate] = useState('')
  const [comparison, setComparison] = useState<ReportComparison | null>(null)

  useEffect(() => {
    let disposed = false
    const refresh = () => {
      api
        .evaluationReports()
        .then((next) => {
          if (!disposed) {
            setListing(next)
            setError(null)
          }
        })
        .catch((caught: unknown) => {
          if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not load evaluation reports')
        })
    }
    refresh()
    const timer = window.setInterval(refresh, pollMilliseconds)
    return () => {
      disposed = true
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (selected === null) return
    let disposed = false
    api
      .evaluationReport(selected)
      .then((next) => { if (!disposed) setView(next) })
      .catch((caught: unknown) => {
        if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not load the report')
      })
    return () => { disposed = true }
  }, [selected])

  const compare = () => {
    setComparison(null)
    api
      .compareReports(baseline, candidate)
      .then(setComparison)
      .catch((caught: unknown) => setError(caught instanceof Error ? caught.message : 'Could not compare reports'))
  }

  const reports = listing?.reports ?? []
  const suiteValues: Record<string, string> = checkpointPath ? { checkpoint_path: checkpointPath } : {}
  const scanValues: Record<string, string> = tokenizedId ? { training_data: tokenizedId } : {}
  return (
    <div className="page-stack">
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      <p className="plan-warning">
        Harness-fixture results only show that the harness works; proxy-scale results measure small
        local checkpoints on named data. No figure here is a capability claim about any EPOR model.
        Code tasks run only from the owner's shell with <code>epor eval suite --allow-code-execution</code>;
        evaluation jobs always skip them.
      </p>
      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Evaluation reports</h2>
            <p>Content-addressed suite reports, verified before they are listed.</p>
          </div>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={() => onNew('eval_suite', suiteValues)}>
              Run suite
            </button>
            <button className="ghost-button" type="button" onClick={() => onNew('eval_contamination', scanValues)}>
              Scan contamination
            </button>
          </div>
        </div>
        {(listing?.integrity_errors.length ?? 0) > 0 && (
          <div className="error-banner" role="alert">
            <strong>Verification failed</strong>
            {listing?.integrity_errors.join('; ')}
          </div>
        )}
        {reports.length === 0 ? (
          <EmptyState>No evaluation report is stored. Train a kernel checkpoint, then run a suite.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table aria-label="Stored evaluation reports">
              <thead>
                <tr>
                  <th>Report</th><th>Suite</th><th>Scope</th><th>Model</th><th>Tasks</th>
                  <th>Primary metrics</th><th><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => (
                  <tr key={report.report_id}>
                    <td><code className="hash" title={report.report_id}>{shortId(report.report_id)}</code></td>
                    <td>{report.suite_id} v{report.suite_version}</td>
                    <td>{report.scope}</td>
                    <td>
                      {report.model_id}
                      {report.checkpoint_step !== null && <small> · step {report.checkpoint_step}</small>}
                    </td>
                    <td className="numeric-cell">
                      {report.task_count}
                      {report.skipped_task_count > 0 && <small> (+{report.skipped_task_count} skipped)</small>}
                      {report.code_executed && <small> · code run</small>}
                    </td>
                    <td>
                      {report.primary_metrics.map((row) => (
                        <div key={row.task_id}>
                          <small>{row.task_id}</small> {row.metric} {formatMetric(row.value, 3)}
                        </div>
                      ))}
                    </td>
                    <td className="action-cell">
                      <button type="button" className="ghost-button" onClick={() => setSelected(report.report_id)}>
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {reports.length >= 2 && (
          <div className="compare-row">
            <label className="form-field">
              <span>Baseline</span>
              <select aria-label="Baseline report" value={baseline} onChange={(event) => setBaseline(event.target.value)}>
                <option value="">Select…</option>
                {reports.map((report) => (
                  <option key={report.report_id} value={report.report_id}>
                    {shortId(report.report_id)} · {report.suite_id} · {report.model_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-field">
              <span>Candidate</span>
              <select aria-label="Candidate report" value={candidate} onChange={(event) => setCandidate(event.target.value)}>
                <option value="">Select…</option>
                {reports.map((report) => (
                  <option key={report.report_id} value={report.report_id}>
                    {shortId(report.report_id)} · {report.suite_id} · {report.model_id}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="ghost-button"
              disabled={baseline === '' || candidate === ''}
              onClick={compare}
            >
              Compare
            </button>
          </div>
        )}
      </section>
      {comparison && <ComparisonPanel comparison={comparison} />}
      {view && <ReportDetail view={view} />}
    </div>
  )
}
