import { EmptyState, StatusPill, shortId } from './shared'
import type { FertilityMetrics, FertilityReport, JobType, TokenizerSummary } from './types'

type StartJob = (type: JobType, values?: Record<string, string>) => void

function ratio(value: number | null, digits = 3): string {
  return value === null ? '—' : value.toFixed(digits)
}

function percent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(2)}%`
}

function MetricsRow({ label, metrics }: { label: string; metrics: FertilityMetrics }) {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td>{metrics.documents.toLocaleString()}</td>
      <td>{metrics.bytes.toLocaleString()}</td>
      <td>{metrics.tokens.toLocaleString()}</td>
      <td>{ratio(metrics.tokens_per_byte)}</td>
      <td>{ratio(metrics.tokens_per_word)}</td>
      <td>{ratio(metrics.bytes_per_token)}</td>
      <td>{percent(metrics.byte_fallback_rate)}</td>
    </tr>
  )
}

function ReportPanel({ report }: { report: FertilityReport }) {
  const passed = report.round_trip_failures === 0
  return (
    <section className="panel table-panel" aria-label={`Fertility report ${shortId(report.report_id)}`}>
      <div className="panel-heading">
        <div>
          <h2>
            {report.dataset_id} · {report.split} split
          </h2>
          <p>
            Tokenizer <code className="hash">{shortId(report.tokenizer_id)}</code> · build{' '}
            <code className="hash">{shortId(report.build_id)}</code> · report{' '}
            <code className="hash">{shortId(report.report_id)}</code>
          </p>
        </div>
        <div className="button-row">
          <StatusPill status={passed ? 'ok' : 'failed'} />
        </div>
      </div>
      {report.in_sample && (
        <div className="error-banner" role="alert">
          <strong>In-sample measurement</strong>
          This split is the tokenizer's own training text. It does not describe held-out fertility.
        </div>
      )}
      {!passed && (
        <div className="error-banner" role="alert">
          <strong>Round-trip gate failed</strong>
          {report.round_trip_failures.toLocaleString()} document(s) did not decode to their exact
          source bytes.
        </div>
      )}
      <p className="plan-warning report-scope">{report.scope}</p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Group</th><th>Docs</th><th>Bytes</th><th>Tokens</th><th>Tokens / byte</th>
              <th>Tokens / word</th><th>Bytes / token</th><th>Byte fallback</th>
            </tr>
          </thead>
          <tbody>
            <MetricsRow label="Overall" metrics={report.overall} />
            {Object.entries(report.by_domain).map(([domain, metrics]) => (
              <MetricsRow key={`domain-${domain}`} label={`Domain: ${domain}`} metrics={metrics} />
            ))}
            {Object.entries(report.by_language).map(([language, metrics]) => (
              <MetricsRow key={`language-${language}`} label={`Language: ${language}`} metrics={metrics} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="panel-footer report-footnote">
        Round-trip failures: {report.round_trip_failures.toLocaleString()} · excluded by removal:{' '}
        {report.excluded_document_count.toLocaleString()} · learned vocabulary{' '}
        {report.learned_vocab_size.toLocaleString()} of {report.vocab_size.toLocaleString()} declared
      </p>
    </section>
  )
}

export function TokenizerPage({
  summary,
  error,
  latestBuilds,
  onNew,
}: {
  summary: TokenizerSummary | null
  error: string | null
  latestBuilds: Record<string, string>
  onNew: StartJob
}) {
  const [latestDataset, latestBuild] = Object.entries(latestBuilds)[0] ?? []
  const trainValues: Record<string, string> =
    latestDataset && latestBuild ? { dataset_id: latestDataset, build_id: latestBuild } : {}
  const tokenizers = summary?.tokenizers ?? []
  const tokenized = summary?.tokenized_builds ?? []
  const reports = summary?.fertility_reports ?? []
  const metrics = [
    ['Tokenizers', tokenizers.length, 'Content-addressed artifacts'],
    ['Tokenized builds', tokenized.length, 'uint16 shards with document boundaries'],
    ['Fertility reports', reports.length, 'Measurements, not validated results'],
    [
      'Round-trip failures',
      reports.reduce((total, report) => total + report.round_trip_failures, 0),
      'Expected to be zero',
    ],
  ] as const
  return (
    <div className="page-stack">
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      <p className="plan-warning">
        Every artifact here was trained and measured on a local proxy-scale build. No target-scale
        tokenizer has been trained; the declared vocabulary is a target, and only the learned size
        reflects what the corpus supported.
      </p>
      <section className="metric-grid" aria-label="Tokenizer summary">
        {metrics.map(([label, value, note]) => (
          <article className="metric-card" key={label}>
            <span>{label}</span>
            <strong>{value.toLocaleString()}</strong>
            <small>{note}</small>
          </article>
        ))}
      </section>
      {(summary?.integrity_errors.length ?? 0) > 0 && (
        <div className="error-banner" role="alert">
          {summary?.integrity_errors.length} tokenizer artifact(s) failed verification and are not
          listed.
        </div>
      )}
      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Tokenizers</h2>
            <p>Byte-level BPE trained on a build's train split only, with 256 frozen reserved IDs.</p>
          </div>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={() => onNew('tokenizer_train', trainValues)}>
              Train
            </button>
          </div>
        </div>
        {tokenizers.length === 0 ? (
          <EmptyState>No tokenizer is stored. Build a dataset, then queue tokenizer training.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Tokenizer</th><th>Trained on</th><th>Declared vocab</th><th>Learned vocab</th>
                  <th>Merges</th><th>Train docs</th><th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {tokenizers.map((item) => {
                  const source = { dataset_id: item.dataset_id, build_id: item.build_id, tokenizer_id: item.tokenizer_id }
                  return (
                    <tr key={item.tokenizer_id}>
                      <td><code className="hash" title={item.tokenizer_id}>{shortId(item.tokenizer_id)}</code></td>
                      <td>{item.dataset_id} <code className="hash" title={item.build_id}>{shortId(item.build_id)}</code></td>
                      <td>{item.declared_vocab_size.toLocaleString()}</td>
                      <td>{item.learned_vocab_size.toLocaleString()}</td>
                      <td>{item.merge_count.toLocaleString()}</td>
                      <td>{item.train_documents.toLocaleString()}</td>
                      <td>
                        <div className="button-row">
                          <button className="ghost-button" type="button" onClick={() => onNew('tokenizer_shard', source)}>Shard</button>
                          <button className="ghost-button" type="button" onClick={() => onNew('tokenizer_evaluate', source)}>Evaluate</button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Tokenized builds</h2>
            <p>Removal tombstones are re-applied at shard time; excluded documents never enter shards.</p>
          </div>
        </div>
        {tokenized.length === 0 ? (
          <EmptyState>No tokenized build is stored.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Tokenized build</th><th>Dataset</th><th>Tokenizer</th><th>Documents</th>
                  <th>Tokens</th><th>Shards</th><th>Excluded</th>
                </tr>
              </thead>
              <tbody>
                {tokenized.map((item) => (
                  <tr key={item.tokenized_id}>
                    <td><code className="hash" title={item.tokenized_id}>{shortId(item.tokenized_id)}</code></td>
                    <td>{item.dataset_id}</td>
                    <td><code className="hash" title={item.tokenizer_id}>{shortId(item.tokenizer_id)}</code></td>
                    <td>{item.total_documents.toLocaleString()}</td>
                    <td>{item.total_tokens.toLocaleString()}</td>
                    <td>{item.shard_count.toLocaleString()}</td>
                    <td>{item.excluded_document_count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {reports.length === 0 ? (
        <section className="panel">
          <EmptyState>No fertility report is stored. Evaluate a tokenizer on a held-out split.</EmptyState>
        </section>
      ) : (
        reports.map((report) => <ReportPanel report={report} key={report.report_id} />)
      )}
    </div>
  )
}
