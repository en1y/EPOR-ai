import { Fragment } from 'react'
import { EmptyState, StatusPill, formatDate, shortId } from './shared'
import type { JobType, KernelRun, TokenizedBuild, TrainingSummary } from './types'

type StartJob = (type: JobType, values?: Record<string, string | boolean>) => void

function loss(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toFixed(4)
}

function count(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toLocaleString()
}

function losses(values: number[]): string {
  return values.length === 0 ? '—' : values.map((value) => value.toFixed(4)).join(' / ')
}

function features(run: KernelRun): string {
  const parts = [run.qk_norm ? 'QK-norm' : 'pre-norm', run.mtp_depth > 0 ? `MTP ×${run.mtp_depth}` : 'no MTP']
  return parts.join(' · ')
}

function topology(run: KernelRun): string {
  const strategy = run.strategy ?? '—'
  return strategy === 'none' ? '1 process' : `${strategy} ×${run.world_size ?? '?'}`
}

function objectiveLabel(run: KernelRun): string {
  if (run.objective === 'dpo') return 'DPO'
  if (run.objective === 'logit-distillation') return `logit distillation ×${run.distill_teachers ?? 0}`
  return 'SFT'
}

/**
 * A run or report path with long hex IDs (sweep points, report nonces) shortened and line
 * breaks allowed only after a separator; the tooltip is the full path.
 */
function Location({ path }: { path: string }) {
  const parts = path.replace(/[0-9a-f]{16,}/g, (hex) => `${shortId(hex)}…`).split('/')
  return (
    <span className="run-location" title={path}>
      {parts.map((part, index) => (
        <Fragment key={index}>
          {part}
          {index < parts.length - 1 && <>/<wbr /></>}
        </Fragment>
      ))}
    </span>
  )
}

/** The tokenized build of the most recent dataset build, else the first one listed. */
export function latestTokenizedBuild(
  tokenized: TokenizedBuild[],
  latestBuilds: Record<string, string>,
): TokenizedBuild | null {
  const latest = new Set(Object.values(latestBuilds))
  return tokenized.find((item) => latest.has(item.build_id)) ?? tokenized[0] ?? null
}

export function KernelRunsPanel({
  summary,
  error,
  tokenized,
  onNew,
}: {
  summary: TrainingSummary | null
  error: string | null
  tokenized: TokenizedBuild | null
  onNew: StartJob
}) {
  const runs = summary?.runs ?? []
  const evaluations = summary?.evaluations ?? []
  const trainValues: Record<string, string | boolean> = tokenized
    ? { tokenized_id: tokenized.tokenized_id, vocab_from_build: true }
    : {}
  const latest = runs[0]
  const metrics = [
    ['Kernel runs', count(runs.length), 'Local CPU, at most 50M parameters'],
    ['Latest main loss', loss(latest?.main_loss), 'Last logged optimizer step'],
    ['Latest validation perplexity', loss(latest?.evaluation?.perplexity), 'From main loss only'],
    ['Evaluation reports', count(evaluations.length), 'Stored kernel evaluations'],
  ] as const
  return (
    <section className="panel table-panel" aria-label="Training kernel">
      <div className="panel-heading">
        <div>
          <h2>Training kernel</h2>
          <p>
            Packed token shards, AdamW with warmup and cosine decay, BF16 autocast, accumulation,
            and exact checkpointed resume. Distributed ddp and fsdp2 runs start from the CLI.
          </p>
        </div>
        <div className="button-row">
          <button
            className="primary-button"
            type="button"
            disabled={tokenized === null}
            title={tokenized === null ? 'Shard a dataset build first' : undefined}
            onClick={() => onNew('kernel_train', trainValues)}
          >
            Train on {tokenized ? `${tokenized.dataset_id} shards` : 'tokenized build'}
          </button>
        </div>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      <p className="plan-warning report-scope">
        {summary?.scope ??
          'Proxy-scale local measurements on named tokenized builds; none is a trained or validated model result.'}{' '}
        Losses compare runs on the same build only. Whether QK-norm or multi-token prediction helps
        is undecided until G3/G5 compares each against its control at matched tokens.
      </p>
      <div className="metric-grid" aria-label="Training kernel summary">
        {metrics.map(([label, value, note]) => (
          <article className="metric-card" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{note}</small>
          </article>
        ))}
      </div>
      {(summary?.integrity_errors.length ?? 0) > 0 && (
        <div className="error-banner" role="alert">
          <strong>Verification failed</strong>
          {summary?.integrity_errors.length} run or report(s) did not verify and are not listed:{' '}
          {summary?.integrity_errors.join('; ')}
        </div>
      )}
      {(summary?.notices?.length ?? 0) > 0 && (
        <p className="plan-warning" role="status">
          {summary?.notices?.join('; ')}
        </p>
      )}
      {runs.length === 0 ? (
        <EmptyState>
          No kernel run is stored. Shard a dataset build, then queue kernel training on it.
        </EmptyState>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Run</th><th>Status</th><th>Step</th><th>Loss</th><th>Main loss</th>
                <th>MTP losses</th><th>Tokens seen</th><th>Tokens / s</th><th>Precision</th>
                <th>Topology</th><th>Features</th><th>Checkpoint</th><th>Validation</th>
                <th><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={`${run.location}-${run.run_id}`}>
                  <td>
                    <code className="hash" title={run.run_id}>{shortId(run.run_id.replace('kernel-', ''))}</code>{' '}
                    <small>{run.source === 'job' ? 'job' : 'CLI'} · <Location path={run.location} /></small>
                    {run.objective && run.objective !== 'pretrain' && (
                      <>
                        {' '}
                        <small className="objective-tag">{objectiveLabel(run)}</small>
                      </>
                    )}
                    <br />
                    <small>{formatDate(run.started_at)}</small>
                  </td>
                  <td><StatusPill status={run.status} /></td>
                  <td className="numeric-cell">
                    {count(run.step)} / {count(run.target_steps)}
                  </td>
                  <td className="numeric-cell">{loss(run.training_loss)}</td>
                  <td className="numeric-cell">{loss(run.main_loss)}</td>
                  <td className="numeric-cell">{losses(run.mtp_losses)}</td>
                  <td className="numeric-cell">{count(run.tokens_seen)}</td>
                  <td className="numeric-cell">
                    {run.tokens_per_second === null ? '—' : Math.round(run.tokens_per_second).toLocaleString()}
                  </td>
                  <td>{run.precision ?? '—'}</td>
                  <td>{topology(run)}</td>
                  <td>{features(run)}</td>
                  <td className="numeric-cell">
                    {run.checkpoint ? (
                      <span title={run.checkpoint.checkpoint_sha256}>step {count(run.checkpoint.step)}</span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="numeric-cell">
                    {run.evaluation
                      ? `${loss(run.evaluation.main_loss)} · ppl ${loss(run.evaluation.perplexity)}`
                      : run.preference_evaluation
                        ? `pref. accuracy ${loss(Number(run.preference_evaluation.accuracy))}`
                        : '—'}
                    {(run.alerts?.length ?? 0) > 0 && <><br /><small>{run.alerts?.length} monitor alert(s)</small></>}
                  </td>
                  <td className="action-cell">
                    {run.source === 'job' && run.checkpoint && (
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => onNew('kernel_evaluate', { checkpoint_path: run.location })}
                      >
                        Evaluate
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {evaluations.length > 0 && (
        <div className="table-scroll">
          <table aria-label="Kernel evaluation reports">
            <thead>
              <tr>
                <th>Report</th><th>Checkpoint step</th><th>Split</th><th>Main loss</th>
                <th>Perplexity</th><th>MTP losses by depth</th><th>Target tokens</th><th>Created</th>
              </tr>
            </thead>
            <tbody>
              {evaluations.map((report) => (
                <tr key={report.location}>
                  <td><code className="hash" title={report.report_sha256}>{shortId(report.report_sha256)}</code> <small><Location path={report.location} /></small></td>
                  <td className="numeric-cell">{count(report.checkpoint_step)}</td>
                  <td>{report.split ?? '—'}{report.split === 'train' ? ' (in-sample)' : ''}</td>
                  <td className="numeric-cell">{loss(report.main_loss)}</td>
                  <td className="numeric-cell">{loss(report.perplexity)}</td>
                  <td className="numeric-cell">{losses(report.mtp_losses)}</td>
                  <td className="numeric-cell">{count(report.target_tokens)}</td>
                  <td>{formatDate(report.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
