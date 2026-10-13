import { useEffect, useState } from 'react'
import { api } from './api'
import { formatMetric } from './EvaluationPage'
import { EmptyState, StatusPill, formatDate, shortId, titleCase } from './shared'
import type { JobType, Ladder, ScalingReport, ScalingReportList, ScalingReportSummary } from './types'

type StartJob = (type: JobType, values?: Record<string, string | boolean>) => void
type Json = Record<string, unknown>

const pollMilliseconds = 5_000

function record(value: unknown): Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Json) : {}
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function number(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : '—'
}

function compact(value: number | null): string {
  if (value === null) return '—'
  const magnitude = Math.abs(value)
  if (magnitude >= 1e15) return value.toExponential(3)
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (magnitude >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4)
}

function headline(report: ScalingReportSummary): string {
  if (report.kind === 'hardware-profile') {
    const mfu = number(report.mfu)
    return `${text(report.model_slug)} · ${compact(number(report.tokens_per_second))} tokens/s · MFU ${
      mfu === null ? 'unknown' : `${(mfu * 100).toFixed(2)}%`
    }`
  }
  if (report.kind === 'scaling-sweep') {
    return `${compact(number(report.completed_points))} completed · ${compact(
      number(report.projection_only_points),
    )} projection-only · ${compact(number(report.failed_points))} failed`
  }
  if (report.kind === 'mixture-ablation') return `${compact(number(report.mixtures))} mixture(s)`
  if (report.kind === 'scaling-fit') {
    return Object.entries(record(report.fit_status))
      .map(([part, status]) => `${titleCase(part)}: ${text(status)}`)
      .join(' · ')
  }
  return `${text(report.target_slug)} · ${compact(number(report.unknown_input_count))} unknown input(s) · approval required`
}

function Warnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null
  return (
    <ul className="plain-list" aria-label="Report warnings">
      {warnings.map((warning) => (
        <li key={warning}>{warning}</li>
      ))}
    </ul>
  )
}

function ProfileDetail({ report }: { report: ScalingReport }) {
  const measured = record(report.measured)
  const mfu = number(measured.mfu)
  const rows: [string, string][] = [
    ['Tokens / second (measured)', compact(number(measured.tokens_per_second))],
    ['Mean step seconds', formatMetric(number(measured.step_seconds_mean))],
    ['Achieved FLOP/s (analytic FLOPs × measured rate)', compact(number(measured.achieved_flops_per_second))],
    ['MFU', mfu === null ? 'unknown — no operator-supplied device peak' : `${(mfu * 100).toFixed(2)}%`],
    ['Process peak RSS (bytes)', compact(number(measured.process_peak_rss_bytes))],
    ['Torch threads', compact(number(measured.torch_threads))],
  ]
  return (
    <div className="table-scroll">
      <table aria-label="Host profile">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td className="numeric-cell">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SweepDetail({ report }: { report: ScalingReport }) {
  const points = list(report.results.points).map(record)
  return (
    <div className="table-scroll">
      <table aria-label="Sweep points">
        <thead>
          <tr>
            <th>Rung</th><th>Status</th><th>Token budget</th><th>Seed</th><th>Steps</th>
            <th>Non-embedding N</th><th>Final main loss</th><th>Validation loss</th>
          </tr>
        </thead>
        <tbody>
          {points.map((point) => (
            <tr key={text(point.point_id)}>
              <td>{text(point.rung)}</td>
              <td><StatusPill status={text(point.status)} /></td>
              <td className="numeric-cell">{compact(number(point.token_budget))}</td>
              <td className="numeric-cell">{compact(number(point.seed))}</td>
              <td className="numeric-cell">{compact(number(point.max_steps))}</td>
              <td className="numeric-cell">{compact(number(point.parameters_non_embedding))}</td>
              <td className="numeric-cell">{formatMetric(number(point.final_main_loss))}</td>
              <td className="numeric-cell">{formatMetric(number(point.validation_main_loss))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MixtureDetail({ report }: { report: ScalingReport }) {
  const mixtures = list(report.results.mixtures).map(record)
  return (
    <div className="table-scroll">
      <table aria-label="Mixture validation losses">
        <thead>
          <tr>
            <th>Mixture</th><th>Overall validation loss</th><th>By domain</th><th>Epochs over subset</th>
          </tr>
        </thead>
        <tbody>
          {mixtures.map((mixture) => {
            const validation = record(mixture.validation)
            const domains = Object.entries(record(validation.by_domain))
            return (
              <tr key={text(mixture.mixture_id)}>
                <td>{text(mixture.mixture_id)}</td>
                <td className="numeric-cell">{formatMetric(number(record(validation.overall).main_loss))}</td>
                <td>
                  {domains.map(([domain, values]) => (
                    <div key={domain}>
                      {domain} {formatMetric(number(record(values).main_loss))}
                    </div>
                  ))}
                </td>
                <td className="numeric-cell">{formatMetric(number(mixture.epochs_over_subset), 2)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function FitDetail({ report }: { report: ScalingReport }) {
  const parts = ['loss_vs_parameters', 'loss_vs_tokens', 'joint'] as const
  return (
    <div className="table-scroll">
      <table aria-label="Scaling fits">
        <thead>
          <tr>
            <th>Fit</th><th>Form</th><th>Status</th><th>Parameters</th><th>RMSE</th>
          </tr>
        </thead>
        <tbody>
          {parts.map((part) => {
            const fit = record(report.results[part])
            const parameters = Object.entries(record(fit.parameters))
            return (
              <tr key={part}>
                <td>{titleCase(part)}</td>
                <td><code>{text(fit.form)}</code></td>
                <td>{text(fit.status)}</td>
                <td>
                  {parameters.length === 0
                    ? '—'
                    : parameters.map(([name, value]) => `${name} ${compact(number(value))}`).join(' · ')}
                </td>
                <td className="numeric-cell">{formatMetric(number(fit.rmse))}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function estimate(value: unknown): string {
  const item = record(value)
  const amount = number(item.value)
  if (amount === null) return 'unknown'
  return typeof item.unit === 'string' ? `${compact(amount)} ${item.unit}` : compact(amount)
}

function ProjectionDetail({ report }: { report: ScalingReport }) {
  const results = report.results
  const approval = record(results.approval)
  const unknown = list(results.unknown_inputs).map(String)
  const target = record(results.target)
  const rows: [string, unknown][] = [
    ['Training tokens', record(results.tokens).training_tokens],
    ['Training FLOPs', record(results.training_flops).total],
    ['Total device-hours', record(results.gpu_hours).total_device_hours],
    ['Wall-clock hours', record(results.gpu_hours).wall_clock_hours],
    ['Training-state memory', record(results.memory).training_state_bytes],
    ['Retained storage', record(results.storage).total_bytes],
    ['Failure reserve fraction', record(results.failure_reserve).fraction],
    ['Total cost with reserve', record(results.cost).total_with_reserve],
    ['Energy', record(results.energy_carbon).energy_kwh],
    ['Expected final loss', record(results.budget_abort_envelope).expected_final_loss],
  ]
  return (
    <>
      <div className="error-banner approval-banner" role="alert">
        <strong>Approval required</strong>
        {text(approval.statement)} Nothing has been provisioned, and this console offers no way to
        launch a target run.
      </div>
      {unknown.length > 0 && (
        <div className="plan-warning report-scope" role="note">
          <strong>Unknown inputs ({unknown.length}).</strong> Every quantity that depends on these is
          reported as unknown: {unknown.join(', ')}
        </div>
      )}
      <p className="report-footnote report-scope">
        Target {text(target.slug)} · local execution: {text(target.local_execution)}
      </p>
      <div className="table-scroll">
        <table aria-label="Projection estimates">
          <thead>
            <tr>
              <th>Quantity</th><th>Estimate</th><th>Evidence label</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, value]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td className="numeric-cell">{estimate(value)}</td>
                <td><span className="estimate-label">{text(record(value).label)}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function ReportDetail({ report }: { report: ScalingReport }) {
  return (
    <section className="panel" aria-label={`Scaling report ${shortId(report.report_id)}`}>
      <div className="panel-heading">
        <div>
          <h2>{titleCase(report.kind)}</h2>
          <p>
            Report <code className="hash" title={report.report_id}>{shortId(report.report_id)}</code> ·
            scope {report.scope}
          </p>
        </div>
        <div>
          <StatusPill status={report.scope} />
        </div>
      </div>
      <p className="plan-warning report-scope">
        <strong>Claim boundary.</strong> {report.claim_boundary}
      </p>
      {report.kind === 'hardware-profile' && <ProfileDetail report={report} />}
      {report.kind === 'scaling-sweep' && <SweepDetail report={report} />}
      {report.kind === 'mixture-ablation' && <MixtureDetail report={report} />}
      {report.kind === 'scaling-fit' && <FitDetail report={report} />}
      {report.kind === 'compute-projection' && <ProjectionDetail report={report} />}
      <Warnings warnings={report.warnings} />
    </section>
  )
}

export function ScalingPage({ onNew, tokenizedId }: { onNew: StartJob; tokenizedId: string | null }) {
  const [ladder, setLadder] = useState<Ladder | null>(null)
  const [listing, setListing] = useState<ScalingReportList | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<ScalingReport | null>(null)

  useEffect(() => {
    let disposed = false
    api
      .ladder()
      .then((next) => { if (!disposed) setLadder(next) })
      .catch((caught: unknown) => {
        if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not load the ladder')
      })
    const refresh = () => {
      api
        .scalingReports()
        .then((next) => { if (!disposed) setListing(next) })
        .catch((caught: unknown) => {
          if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not load scaling reports')
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
    const [reportId, contentSha256] = selected.split(':')
    api
      .scalingReport(reportId)
      .then((reports) => {
        if (disposed) return
        setDetail(reports.find((item) => item.summary.content_sha256 === contentSha256) ?? reports[0] ?? null)
      })
      .catch((caught: unknown) => {
        if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not load the report')
      })
    return () => { disposed = true }
  }, [selected])

  const template = tokenizedId ? JSON.stringify({ tokenized_id: tokenizedId }) : ''
  const reports = listing?.reports ?? []
  return (
    <div className="page-stack">
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      <p className="plan-warning">
        Sweeps and mixtures are proxy-scale measurements of small local models; profiles are
        measurements of this host only; projections are arithmetic over operator-supplied inputs.
        Nothing here provisions compute, and no target-scale run has been approved. Projections are
        made on the CLI with <code>epor scale project</code> and are never an approval.
      </p>
      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Model-size ladder</h2>
            <p>{ladder?.note ?? 'Architecture contracts; none is a trained model.'}</p>
          </div>
        </div>
        <div className="table-scroll">
          <table aria-label="Ladder rungs">
            <thead>
              <tr>
                <th>Rung</th><th>Non-embedding</th><th>Total</th><th>Vocabulary</th><th>Execution</th><th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {(ladder?.rungs ?? []).map((rung) => (
                <tr key={rung.slug}>
                  <td>{rung.slug}</td>
                  <td className="numeric-cell">{compact(rung.non_embedding_parameters)}</td>
                  <td className="numeric-cell">{compact(rung.total_parameters)}</td>
                  <td className="numeric-cell">{rung.vocab_size.toLocaleString()}</td>
                  <td><StatusPill status={rung.execution} /></td>
                  <td className="reason-cell"><small>{rung.reason}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Scaling reports</h2>
            <p>Profiles, sweeps, mixture ablations, fits, and projections, verified before listing.</p>
          </div>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={() => onNew('scale_profile', tokenizedId ? { vocab_from_build: tokenizedId } : {})}>
              Profile host
            </button>
            <button className="ghost-button" type="button" onClick={() => onNew('scale_sweep', template ? { training: template } : {})}>
              Sweep
            </button>
            <button className="ghost-button" type="button" onClick={() => onNew('scale_mixture', template ? { training: template } : {})}>
              Mixture ablation
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
          <EmptyState>No scaling report is stored. Profile this host or run a small sweep.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table aria-label="Stored scaling reports">
              <thead>
                <tr>
                  <th>Report</th><th>Kind</th><th>Scope</th><th>Summary</th><th>Warnings</th><th>Recorded</th>
                  <th><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => (
                  <tr key={`${report.report_id}:${report.content_sha256}`}>
                    <td><code className="hash" title={report.report_id}>{shortId(report.report_id)}</code></td>
                    <td>{titleCase(report.kind)}</td>
                    <td>{report.scope}</td>
                    <td>{headline(report)}</td>
                    <td className="numeric-cell">{report.warning_count}</td>
                    <td>{formatDate(report.recorded_at)}</td>
                    <td className="action-cell">
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => setSelected(`${report.report_id}:${report.content_sha256}`)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {detail && <ReportDetail report={detail} />}
    </div>
  )
}
