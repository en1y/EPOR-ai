import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { KernelRunsPanel, latestTokenizedBuild } from './KernelRuns'
import type { KernelRun, TokenizedBuild, TrainingSummary } from './types'

const tokenized: TokenizedBuild = {
  tokenized_id: 'd'.repeat(64),
  tokenizer_id: 'a'.repeat(64),
  dataset_id: 'fixture-v1',
  build_id: 'b'.repeat(64),
  max_shard_tokens: 16777216,
  total_tokens: 98765,
  total_documents: 6,
  excluded_document_count: 0,
  applied_tombstone_ids: [],
  shard_count: 3,
  splits: {},
}

const run: KernelRun = {
  run_id: 'kernel-0123456789abcdef',
  lineage_id: 'lineage-1',
  parent_run_id: null,
  source: 'job',
  location: 'job-1/kernel',
  status: 'succeeded',
  termination_reason: 'completed',
  started_at: '2026-10-13T12:00:00Z',
  finished_at: '2026-10-13T12:01:00Z',
  model_id: 'epor-tiny',
  config_sha256: 'c'.repeat(64),
  qk_norm: true,
  mtp_depth: 1,
  precision: 'bf16',
  strategy: 'none',
  world_size: 1,
  tokenized_id: tokenized.tokenized_id,
  seq_len: 128,
  tokens_per_batch: 2048,
  start_step: 0,
  target_steps: 60,
  step: 60,
  training_loss: 8.0662,
  main_loss: 6.2053,
  mtp_losses: [6.2034],
  learning_rate: 0.0003,
  tokens_seen: 122880,
  tokens_per_second: 2570.7,
  evaluation: {
    split: 'validation',
    main_loss: 6.7019,
    perplexity: 813.99,
    mtp_losses: [6.7295],
    target_tokens: 8192,
    steps: 4,
  },
  checkpoint: { name: 'step-00000060', step: 60, world_size: 1, checkpoint_sha256: 'e'.repeat(64) },
  manifest_sha256: 'f'.repeat(64),
}

function summary(overrides: Partial<TrainingSummary> = {}): TrainingSummary {
  return {
    schema_version: 1,
    scope: 'Proxy-scale local measurements on named tokenized builds; none is a trained or validated model result.',
    runs: [run],
    evaluations: [{
      location: 'job-2/kernel-eval/step-00000060-1.json',
      report_sha256: '9'.repeat(64),
      checkpoint_step: 60,
      model_id: 'epor-tiny',
      tokenized_id: tokenized.tokenized_id,
      split: 'test',
      main_loss: 6.2307,
      perplexity: 508.09,
      mtp_losses: [6.25],
      target_tokens: 62324,
      steps: 31,
      created_at: '2026-10-13T12:02:00Z',
    }],
    integrity_errors: [],
    ...overrides,
  }
}

describe('training kernel panel', () => {
  it('shows proxy-scale run measurements, features, and checkpoint state', () => {
    render(<KernelRunsPanel summary={summary()} error={null} tokenized={tokenized} onNew={vi.fn()} />)

    expect(screen.getByText(/Proxy-scale local measurements/)).toBeInTheDocument()
    expect(screen.getByText(/undecided until G3\/G5/)).toBeInTheDocument()
    const row = screen.getByText('job · job-1/kernel').closest('tr')!
    expect(within(row).getByText('60 / 60')).toBeInTheDocument()
    expect(within(row).getByText('6.2053')).toBeInTheDocument()
    expect(within(row).getByText('6.2034')).toBeInTheDocument()
    expect(within(row).getByText('122,880')).toBeInTheDocument()
    expect(within(row).getByText('2,571')).toBeInTheDocument()
    expect(within(row).getByText('QK-norm · MTP ×1')).toBeInTheDocument()
    expect(within(row).getByText('1 process')).toBeInTheDocument()
    expect(within(row).getByText('step 60')).toBeInTheDocument()
    expect(within(row).getByText('6.7019 · ppl 813.9900')).toBeInTheDocument()

    const reports = screen.getByRole('table', { name: 'Kernel evaluation reports' })
    expect(within(reports).getByText('508.0900')).toBeInTheDocument()
    expect(within(reports).getByText('62,324')).toBeInTheDocument()
  })

  it('prefills training from the tokenized build and evaluation from a job run', async () => {
    const user = userEvent.setup()
    const onNew = vi.fn()
    render(<KernelRunsPanel summary={summary()} error={null} tokenized={tokenized} onNew={onNew} />)

    await user.click(screen.getByRole('button', { name: 'Train on fixture-v1 shards' }))
    expect(onNew).toHaveBeenCalledWith('kernel_train', {
      tokenized_id: tokenized.tokenized_id,
      vocab_from_build: true,
    })
    await user.click(screen.getByRole('button', { name: 'Evaluate' }))
    expect(onNew).toHaveBeenCalledWith('kernel_evaluate', { checkpoint_path: 'job-1/kernel' })
  })

  it('offers no evaluation for CLI runs and no training without shards', () => {
    render(
      <KernelRunsPanel
        summary={summary({ runs: [{ ...run, source: 'cli', location: 'runs/fp32', strategy: 'fsdp2', world_size: 2 }] })}
        error={null}
        tokenized={null}
        onNew={vi.fn()}
      />,
    )
    expect(screen.getByText('fsdp2 ×2')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Evaluate' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Train on tokenized build/ })).toBeDisabled()
  })

  it('reports runs that failed verification instead of listing them', () => {
    render(
      <KernelRunsPanel
        summary={summary({ runs: [], evaluations: [], integrity_errors: ['job run job-1/kernel: checkpoint step-00000002 trainer state does not match its digest'] })}
        error={null}
        tokenized={tokenized}
        onNew={vi.fn()}
      />,
    )
    expect(screen.getByRole('alert')).toHaveTextContent('did not verify')
    expect(screen.getByText(/No kernel run is stored/)).toBeInTheDocument()
  })

  it('picks the tokenized build of the latest dataset build', () => {
    const other = { ...tokenized, tokenized_id: '1'.repeat(64), build_id: '2'.repeat(64) }
    expect(latestTokenizedBuild([tokenized, other], { 'fixture-v1': other.build_id })).toBe(other)
    expect(latestTokenizedBuild([tokenized, other], {})).toBe(tokenized)
    expect(latestTokenizedBuild([], {})).toBeNull()
  })
})
