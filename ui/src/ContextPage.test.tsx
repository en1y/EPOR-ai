import { render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ContextPage } from './ContextPage'
import { api } from './api'
import type { ContextSummary } from './types'

const sha = 'c'.repeat(64)

const summary: ContextSummary = {
  schema_version: 1,
  scope: 'Proxy-scale and host measurements.',
  runs: [
    {
      location: 'runs/context/extend',
      source: 'cli',
      run_id: 'run',
      status: 'succeeded',
      model_id: 'epor-tiny',
      started_at: '2026-10-14T10:00:00Z',
      configured: 4096,
      trained: 2048,
      validated: 0,
      rope_scaling: { type: 'yarn', factor: 4, original_max_context: 512 },
      sliding_window: 0,
      local_layers_per_global: 0,
      attention_variant: 'gqa',
      sparse_attention: false,
      extension_of: sha,
      checkpoint_sha256: sha,
    },
  ],
  g6: [
    {
      report_id: 'd'.repeat(64),
      model_id: 'epor-tiny',
      checkpoint_sha256: sha,
      checkpoint_step: 40,
      scope: 'proxy-scale',
      lengths: [256],
      retrieval: { passkey: { '256': { beginning: 0.5, middle: 0.25, end: null } } },
      position_loss: { '256': [5.1, 5.0] },
      injection_followed: 0.5,
      claim_boundary: 'Proxy-scale.',
    },
  ],
  certificates: [],
  profile: [
    {
      report_id: 'e'.repeat(64),
      content_sha256: 'f'.repeat(64),
      scope: 'host-measurement',
      inputs: { quantize: 'int8' },
      results: { lengths: [{ length: 512, status: 'ok', kv_cache_bytes: 2048 }, { length: 32768, status: 'out-of-memory', kv_cache_bytes: null }] },
      measured: { lengths: [{ ttft_seconds: 0.022, decode_tokens_per_second: 916, peak_rss_bytes: 400 * 1024 * 1024 }, {}] },
      warnings: [],
    },
  ],
  feasibility: [],
  sweep: [],
  projection: [],
  integrity_errors: [],
}

afterEach(() => vi.restoreAllMocks())

describe('ContextPage', () => {
  it('keeps configured, trained, and validated context as separate columns', async () => {
    vi.spyOn(api, 'context').mockResolvedValue(summary)
    render(<ContextPage onNew={() => undefined} />)
    const runs = await screen.findByRole('table', { name: 'Context runs' })
    const row = within(runs).getAllByRole('row')[1]
    expect(within(row).getByText('4,096')).toBeInTheDocument()
    expect(within(row).getByText('2,048')).toBeInTheDocument()
    expect(within(row).getByText('0 · not validated')).toBeInTheDocument()
    expect(within(row).getByText(/yarn ×4\.0 from 512/)).toBeInTheDocument()
  })

  it('shows retrieval by position and host measurements, including out-of-memory', async () => {
    vi.spyOn(api, 'context').mockResolvedValue(summary)
    render(<ContextPage onNew={() => undefined} />)
    const heatmap = await screen.findByRole('table', { name: /G6 retrieval by position/ })
    expect(within(heatmap).getByText('0.50')).toBeInTheDocument()
    expect(within(heatmap).getByText('—')).toBeInTheDocument()
    const profiles = screen.getByRole('table', { name: 'Host profiles by length' })
    expect(within(profiles).getByText('0.022 s')).toBeInTheDocument()
    expect(within(profiles).getByText('400.0 MiB')).toBeInTheDocument()
    expect(within(profiles).getByText('out-of-memory')).toBeInTheDocument()
  })
})
