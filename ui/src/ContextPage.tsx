import { useEffect, useState } from 'react'
import { api } from './api'
import { EmptyState, StatusPill, formatDate, shortId, titleCase } from './shared'
import type { ContextScalingReport, ContextSummary, G6View, JobType } from './types'

type StartJob = (type: JobType, values?: Record<string, string | boolean>) => void

const pollMilliseconds = 5_000
const buckets = ['beginning', 'middle', 'end'] as const

function count(value: unknown): string {
  return typeof value === 'number' ? value.toLocaleString() : '—'
}

function fixed(value: unknown, digits = 3): string {
  return typeof value === 'number' ? value.toFixed(digits) : '—'
}

function bytes(value: unknown): string {
  if (typeof value !== 'number') return '—'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let scaled = value
  let unit = 0
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024
    unit += 1
  }
  return `${scaled.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
}

function list(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(record) : []
}

/** A retrieval cell: the value as text, its strength as a background the colour-blind can still read by number. */
function HeatCell({ value }: { value: number | null | undefined }) {
  const known = typeof value === 'number'
  return (
    <td
      className="numeric-cell heat-cell"
      style={known ? { backgroundColor: `color-mix(in srgb, var(--accent) ${Math.round(value * 45)}%, transparent)` } : undefined}
    >
      {known ? value.toFixed(2) : '—'}
    </td>
  )
}

function ContextRuns({ summary }: { summary: ContextSummary | null }) {
  const runs = summary?.runs ?? []
  return (
    <section className="panel table-panel" aria-labelledby="context-runs-heading">
      <div className="panel-heading">
        <div>
          <h2 id="context-runs-heading">Context of each run</h2>
          <p>
            Configured is what the code accepts, trained is the longest length the lineage trained at, and
            validated rises only with a passing G6 certificate. They are different numbers on purpose.
          </p>
        </div>
      </div>
      {runs.length === 0 ? (
        <EmptyState>No context run is stored. Train a kernel run, then extend it.</EmptyState>
      ) : (
        <div className="table-scroll">
          <table aria-label="Context runs">
            <thead>
              <tr>
                <th>Run</th><th>Status</th><th>Configured</th><th>Trained</th><th>Validated</th>
                <th>Positional</th><th>Attention</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={`${run.source}-${run.location}`}>
                  <td>
                    {run.model_id ?? '—'} <small>{run.source === 'job' ? 'job' : 'CLI'} · {run.location}</small>
                    {run.extension_of && <><br /><small>extends config {shortId(run.extension_of)}</small></>}
                  </td>
                  <td><StatusPill status={run.status ?? 'unknown'} /></td>
                  <td className="numeric-cell">{count(run.configured)}</td>
                  <td className="numeric-cell">{count(run.trained)}</td>
                  <td className="numeric-cell">
                    {run.validated ? count(run.validated) : <span title="No G6 certificate">0 · not validated</span>}
                  </td>
                  <td>
                    {run.rope_scaling
                      ? `${run.rope_scaling.type} ×${fixed(run.rope_scaling.factor, 1)} from ${count(run.rope_scaling.original_max_context)}`
                      : 'RoPE, unscaled'}
                  </td>
                  <td>
                    {run.attention_variant === 'mla' ? 'MLA' : 'GQA'}
                    {run.local_layers_per_global
                      ? ` · ${run.local_layers_per_global}:1 local, window ${count(run.sliding_window)}`
                      : ' · full'}
                    {run.sparse_attention ? ' · sparse retrofit' : ''}
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

function G6Report({ view }: { view: G6View }) {
  const families = Object.keys(view.retrieval)
  return (
    <article className="g6-report">
      <h3>
        {view.model_id} <small>step {view.checkpoint_step ?? '—'} · report <code className="hash">{shortId(view.report_id)}</code></small>
      </h3>
      <div className="table-scroll">
        <table aria-label={`G6 retrieval by position for ${shortId(view.report_id)}`}>
          <thead>
            <tr>
              <th>Task</th><th>Length</th>
              {buckets.map((bucket) => <th key={bucket}>{titleCase(bucket)}</th>)}
            </tr>
          </thead>
          <tbody>
            {families.flatMap((family) =>
              view.lengths.map((length) => (
                <tr key={`${family}-${length}`}>
                  <td>{titleCase(family)}</td>
                  <td className="numeric-cell">{count(length)}</td>
                  {buckets.map((bucket) => (
                    <HeatCell key={bucket} value={view.retrieval[family]?.[String(length)]?.[bucket]} />
                  ))}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p className="report-footnote">
        Injection followed: {fixed(view.injection_followed, 2)} · position loss by bucket:{' '}
        {Object.entries(view.position_loss)
          .map(([length, losses]) => `${length}: ${losses.map((loss) => fixed(loss, 2)).join(' / ')}`)
          .join(' · ') || '—'}
      </p>
    </article>
  )
}

function Profiles({ reports }: { reports: ContextScalingReport[] }) {
  if (reports.length === 0) return <EmptyState>No host profile is stored. Run <code>epor context profile</code>.</EmptyState>
  return (
    <div className="table-scroll">
      <table aria-label="Host profiles by length">
        <thead>
          <tr>
            <th>Report</th><th>Length</th><th>Status</th><th>TTFT</th><th>Decode / s</th>
            <th>Peak RSS</th><th>KV cache</th>
          </tr>
        </thead>
        <tbody>
          {reports.flatMap((report) => {
            const content = list(record(report.results).lengths)
            const measured = list(record(report.measured).lengths)
            return content.map((row, index) => (
              <tr key={`${report.content_sha256}-${index}`}>
                <td><code className="hash">{shortId(report.report_id)}</code> <small>{String(record(report.inputs).quantize ?? '')}</small></td>
                <td className="numeric-cell">{count(row.length)}</td>
                <td><StatusPill status={row.status === 'ok' ? 'succeeded' : 'failed'} /> <small>{String(row.status)}</small></td>
                <td className="numeric-cell">{typeof measured[index]?.ttft_seconds === 'number' ? `${fixed(measured[index].ttft_seconds, 3)} s` : '—'}</td>
                <td className="numeric-cell">{fixed(measured[index]?.decode_tokens_per_second, 1)}</td>
                <td className="numeric-cell">{bytes(measured[index]?.peak_rss_bytes)}</td>
                <td className="numeric-cell">{bytes(row.kv_cache_bytes)}</td>
              </tr>
            ))
          })}
        </tbody>
      </table>
    </div>
  )
}

function Feasibility({ reports }: { reports: ContextScalingReport[] }) {
  if (reports.length === 0) return <EmptyState>No feasibility search is stored. Run <code>epor context feasibility</code>.</EmptyState>
  return (
    <div className="table-scroll">
      <table aria-label="Feasible maximum length per variant">
        <thead><tr><th>Report</th><th>Variant</th><th>Train step fits up to</th><th>Prefill fits up to</th></tr></thead>
        <tbody>
          {reports.flatMap((report) =>
            Object.entries(record(report.results)).map(([variant, value]) => {
              const row = record(value)
              return (
                <tr key={`${report.content_sha256}-${variant}`}>
                  <td><code className="hash">{shortId(report.report_id)}</code></td>
                  <td>{titleCase(variant)}</td>
                  <td className="numeric-cell">{count(record(row.train).max_ok_length)}</td>
                  <td className="numeric-cell">{count(record(row.inference).max_ok_length)}</td>
                </tr>
              )
            }),
          )}
        </tbody>
      </table>
    </div>
  )
}

function Sweeps({ reports }: { reports: ContextScalingReport[] }) {
  if (reports.length === 0) return <EmptyState>No context sweep is stored. Run <code>epor context sweep</code>.</EmptyState>
  return (
    <div className="table-scroll">
      <table aria-label="Context sweeps">
        <thead>
          <tr>
            <th>Axis</th><th>Point</th><th>KV bytes / token</th>
            <th>Passkey B / M / E</th><th>Loss</th>
          </tr>
        </thead>
        <tbody>
          {reports.flatMap((report) => {
            const results = record(report.results)
            const rows = [record(results.control), ...list(results.points)]
            return rows.map((row) => {
              const passkey = record(record(row.retrieval).passkey)
              return (
                <tr key={`${report.report_id}-${String(row.name)}`}>
                  <td>{titleCase(String(record(report.inputs).axis ?? ''))}</td>
                  <td>{String(row.name)}</td>
                  <td className="numeric-cell">{fixed(row.kv_bytes_per_token_bf16, 0)}</td>
                  <td className="numeric-cell">{buckets.map((bucket) => fixed(passkey[bucket], 2)).join(' / ')}</td>
                  <td className="numeric-cell">{fixed(row.validation_loss ?? row.final_loss, 3)}</td>
                </tr>
              )
            })
          })}
        </tbody>
      </table>
    </div>
  )
}

function Projections({ reports }: { reports: ContextScalingReport[] }) {
  if (reports.length === 0) return <EmptyState>No context projection is stored. Run <code>epor context project</code>.</EmptyState>
  return (
    <div className="table-scroll">
      <table aria-label="Context projections">
        <thead>
          <tr><th>Target</th><th>Length</th><th>KV cache (bf16)</th><th>Local/global (bf16)</th><th>Local/global (int8)</th><th>Rule</th></tr>
        </thead>
        <tbody>
          {reports.flatMap((report) =>
            list(record(report.results).stages).map((stage) => {
              const cache = record(stage.kv_cache)
              const configured = record(cache['as-configured'])
              const local = record(cache['interleaved-local-global'])
              return (
                <tr key={`${report.report_id}-${String(stage.length)}`}>
                  <td>{String(record(report.inputs).target_model_id ?? '—')}</td>
                  <td className="numeric-cell">{count(stage.length)}</td>
                  <td className="numeric-cell">{bytes(configured.bf16_bytes)}</td>
                  <td className="numeric-cell">{bytes(local.bf16_bytes)}</td>
                  <td className="numeric-cell">{bytes(local.int8_bytes)}</td>
                  <td><small>{String(stage.stage_rule)}</small></td>
                </tr>
              )
            }),
          )}
        </tbody>
      </table>
    </div>
  )
}

export function ContextPage({ onNew }: { onNew: StartJob }) {
  const [summary, setSummary] = useState<ContextSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let disposed = false
    const refresh = () => {
      api
        .context()
        .then((next) => { if (!disposed) { setSummary(next); setError(null) } })
        .catch((caught: unknown) => { if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not load context state') })
    }
    refresh()
    const timer = window.setInterval(refresh, pollMilliseconds)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [])

  return (
    <div className="page-stack">
      {error && <div className="error-banner" role="alert">{error}</div>}
      <p className="plan-warning">
        {summary?.scope ?? 'Configured, trained, and validated context are different numbers.'} Accepting a long
        input never raises the validated context, and 32K/128K/256K appear here only as projections.
      </p>
      {(summary?.integrity_errors.length ?? 0) > 0 && (
        <div className="error-banner" role="alert"><strong>Verification failed</strong> {summary?.integrity_errors.join('; ')}</div>
      )}
      <ContextRuns summary={summary} />
      <section className="panel" aria-labelledby="g6-heading">
        <div className="panel-heading">
          <div>
            <h2 id="g6-heading">G6 retrieval by position</h2>
            <p>Generated passkey, multi-key, multi-document, long-code, and injection items at the beginning, middle, and end. Chance is 0.25.</p>
          </div>
          <div className="button-row">
            <button type="button" className="ghost-button" onClick={() => onNew('context_evaluate')}>Run G6</button>
          </div>
        </div>
        {(summary?.g6.length ?? 0) === 0 ? (
          <EmptyState>No G6 report is stored.</EmptyState>
        ) : (
          summary?.g6.map((view) => <G6Report key={view.report_id} view={view} />)
        )}
      </section>
      <section className="panel table-panel" aria-labelledby="host-heading">
        <div className="panel-heading">
          <div>
            <h2 id="host-heading">Host measurements</h2>
            <p>Time to first token, decode speed, peak RSS, KV cache, and out-of-memory behaviour, each in a bounded subprocess on this machine only.</p>
          </div>
          <div className="button-row">
            <button type="button" className="ghost-button" onClick={() => onNew('context_profile')}>Profile</button>
          </div>
        </div>
        <Profiles reports={summary?.profile ?? []} />
        <Feasibility reports={summary?.feasibility ?? []} />
      </section>
      <section className="panel table-panel" aria-labelledby="sweeps-heading">
        <div className="panel-heading">
          <div>
            <h2 id="sweeps-heading">Sweeps against full attention</h2>
            <p>Window, local:global ratio, and split RoPE base, judged on KV bytes and position-wise retrieval rather than perplexity alone.</p>
          </div>
        </div>
        <Sweeps reports={summary?.sweep ?? []} />
      </section>
      <section className="panel table-panel" aria-labelledby="projection-heading">
        <div className="panel-heading">
          <div>
            <h2 id="projection-heading">Projected stages</h2>
            <p>Analytic KV cache for the progressive recipe from operator inputs. Not trained, not validated, and not an approval.</p>
          </div>
        </div>
        <Projections reports={summary?.projection ?? []} />
      </section>
      {(summary?.certificates.length ?? 0) > 0 && (
        <section className="panel table-panel" aria-labelledby="certificates-heading">
          <div className="panel-heading"><div><h2 id="certificates-heading">Validated-context certificates</h2></div></div>
          <div className="table-scroll">
            <table aria-label="Certificates">
              <thead><tr><th>Checkpoint</th><th>Validated length</th><th>G6 report</th><th>Recorded</th></tr></thead>
              <tbody>
                {summary?.certificates.map((item) => (
                  <tr key={item.certificate_id}>
                    <td><code className="hash">{shortId(item.checkpoint_sha256)}</code></td>
                    <td className="numeric-cell">{count(item.length)}</td>
                    <td><code className="hash">{shortId(item.g6_report_id)}</code></td>
                    <td>{formatDate(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
