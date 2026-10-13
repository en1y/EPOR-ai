import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { TokenizerPage } from './TokenizerPage'
import type { FertilityMetrics, TokenizerSummary } from './types'

const metrics: FertilityMetrics = {
  documents: 2,
  bytes: 400,
  characters: 400,
  words: 80,
  tokens: 120,
  byte_fallback_tokens: 3,
  tokens_per_byte: 0.3,
  tokens_per_word: 1.5,
  bytes_per_token: 3.333333,
  byte_fallback_rate: 0.025,
}

const tokenizerId = 'a'.repeat(64)
const buildId = 'b'.repeat(64)

function summary(overrides: Partial<TokenizerSummary['fertility_reports'][number]> = {}): TokenizerSummary {
  return {
    schema_version: 1,
    tokenizers: [{
      tokenizer_id: tokenizerId,
      tokenizer_version: 'epor-bpe-v1',
      reserved_spec_version: 'epor-reserved-v1',
      declared_vocab_size: 65536,
      learned_vocab_size: 812,
      merge_count: 300,
      dataset_id: 'fixture-v1',
      build_id: buildId,
      train_split_sha256: 'c'.repeat(64),
      train_documents: 4,
      train_bytes: 2048,
      min_frequency: 2,
      library_version: '0.22.2',
    }],
    tokenized_builds: [{
      tokenized_id: 'd'.repeat(64),
      tokenizer_id: tokenizerId,
      dataset_id: 'fixture-v1',
      build_id: buildId,
      max_shard_tokens: 16777216,
      total_tokens: 98765,
      total_documents: 6,
      excluded_document_count: 1,
      applied_tombstone_ids: ['e'.repeat(64)],
      shard_count: 3,
      splits: {},
    }],
    fertility_reports: [{
      report_id: 'f'.repeat(64),
      schema_version: 1,
      kind: 'tokenizer-fertility',
      scope: 'Measured on the named build split only. Values describe this corpus and are not target-scale results.',
      tokenizer_id: tokenizerId,
      tokenizer_json_sha256: '1'.repeat(64),
      reserved_spec_sha256: '2'.repeat(64),
      vocab_size: 65536,
      learned_vocab_size: 812,
      dataset_id: 'fixture-v1',
      build_id: buildId,
      split: 'validation',
      split_sha256: '3'.repeat(64),
      in_sample: false,
      excluded_document_count: 0,
      applied_tombstone_ids: [],
      round_trip_failures: 0,
      overall: metrics,
      by_domain: { code: metrics, general: metrics },
      by_language: { en: metrics },
      ...overrides,
    }],
    integrity_errors: [],
  }
}

describe('Tokenizer page', () => {
  it('keeps declared and learned vocabulary apart and shows per-group fertility', () => {
    render(<TokenizerPage summary={summary()} error={null} latestBuilds={{}} onNew={vi.fn()} />)

    const [tokenizers] = screen.getAllByRole('table')
    const row = within(tokenizers).getAllByRole('row')[1]
    expect(within(row).getByText('65,536')).toBeInTheDocument()
    expect(within(row).getByText('812')).toBeInTheDocument()
    expect(screen.getByText('98,765')).toBeInTheDocument()
    expect(screen.getByText('Domain: code')).toBeInTheDocument()
    expect(screen.getByText('Language: en')).toBeInTheDocument()
    expect(screen.getByText(/not target-scale results/)).toBeInTheDocument()
    expect(screen.getByText(/No target-scale tokenizer has been trained/)).toBeInTheDocument()
    expect(screen.queryByText('In-sample measurement')).not.toBeInTheDocument()
    expect(screen.queryByText('Round-trip gate failed')).not.toBeInTheDocument()
  })

  it('flags in-sample measurements and failed round-trip gates', () => {
    render(
      <TokenizerPage
        summary={summary({ in_sample: true, split: 'train', round_trip_failures: 2 })}
        error={null}
        latestBuilds={{}}
        onNew={vi.fn()}
      />,
    )
    expect(screen.getByText('In-sample measurement')).toBeInTheDocument()
    expect(screen.getByText('Round-trip gate failed')).toBeInTheDocument()
    expect(screen.getByText('failed', { selector: '.status-pill' })).toBeInTheDocument()
  })

  it('queues typed tokenizer jobs prefilled from stored artifacts', async () => {
    const user = userEvent.setup()
    const onNew = vi.fn()
    render(
      <TokenizerPage
        summary={summary()}
        error={null}
        latestBuilds={{ 'fixture-v1': buildId }}
        onNew={onNew}
      />,
    )
    await user.click(screen.getByRole('button', { name: 'Train' }))
    expect(onNew).toHaveBeenLastCalledWith('tokenizer_train', { dataset_id: 'fixture-v1', build_id: buildId })
    const source = { dataset_id: 'fixture-v1', build_id: buildId, tokenizer_id: tokenizerId }
    await user.click(screen.getByRole('button', { name: 'Shard' }))
    expect(onNew).toHaveBeenLastCalledWith('tokenizer_shard', source)
    await user.click(screen.getByRole('button', { name: 'Evaluate' }))
    expect(onNew).toHaveBeenLastCalledWith('tokenizer_evaluate', source)
  })

  it('reports an unavailable tokenizer API and empty stores without crashing', () => {
    render(
      <TokenizerPage
        summary={null}
        error="tokenizer support requires the optional data dependencies"
        latestBuilds={{}}
        onNew={vi.fn()}
      />,
    )
    expect(screen.getByRole('alert')).toHaveTextContent('optional data dependencies')
    expect(screen.getByText(/No tokenizer is stored/)).toBeInTheDocument()
  })
})
