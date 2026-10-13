import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ScalingPage } from './ScalingPage'
import { api } from './api'
import type { Ladder, ScalingReport, ScalingReportList } from './types'

const ladder: Ladder = {
  note: 'Rung files are architecture contracts, not trained models.',
  rungs: [
    {
      slug: 'epor-ladder-10m',
      config_sha256: 'a'.repeat(64),
      nominal_non_embedding: 10_000_000,
      total_parameters: 35_500_000,
      non_embedding_parameters: 10_100_000,
      vocab_size: 65_536,
      execution: 'local',
      reason: 'within the local training kernel bound',
    },
    {
      slug: 'epor-ladder-1b',
      config_sha256: 'b'.repeat(64),
      nominal_non_embedding: 1_000_000_000,
      total_parameters: 1_134_000_000,
      non_embedding_parameters: 1_000_000_000,
      vocab_size: 65_536,
      execution: 'projection-only',
      reason: 'exceeds the local kernel bound; projection only',
    },
  ],
}

const base = {
  claim_boundary: 'Projection from operator-supplied inputs. Not an approval to provision compute.',
  warning_count: 2,
  recorded_at: null,
}

const listing: ScalingReportList = {
  schema_version: 1,
  integrity_errors: [],
  reports: [
    {
      ...base,
      report_id: 'c'.repeat(64),
      content_sha256: 'd'.repeat(64),
      kind: 'hardware-profile',
      scope: 'host-measurement',
      recorded_at: '2026-10-13T12:00:00Z',
      model_slug: 'epor-tiny',
      tokens_per_second: 2570.7,
      mfu: null,
      mfu_status: 'unknown',
    },
    {
      ...base,
      report_id: 'e'.repeat(64),
      content_sha256: 'f'.repeat(64),
      kind: 'compute-projection',
      scope: 'projection',
      target_slug: 'epor-ladder-1b',
      unknown_input_count: 3,
      approval_required: true,
    },
  ],
}

const projection: ScalingReport = {
  report_id: 'e'.repeat(64),
  kind: 'compute-projection',
  scope: 'projection',
  claim_boundary: base.claim_boundary,
  inputs: {},
  warnings: ['Projection only: nothing has been provisioned.'],
  measured: null,
  summary: listing.reports[1],
  results: {
    target: { slug: 'epor-ladder-1b', local_execution: 'projection-only' },
    tokens: { training_tokens: { value: 2e10, unit: 'tokens', label: 'hypothesis' } },
    training_flops: { total: { value: null, unit: 'FLOP', label: 'unknown' } },
    cost: { total_with_reserve: { value: null, unit: 'currency', label: 'unknown' } },
    unknown_inputs: ['hardware.peak_flops_per_device', 'prices.price_per_device_hour', 'run.sequence_length'],
    approval: {
      required: true,
      approval_id: null,
      statement: 'A projection is not authorization. A named human must approve the exact immutable proposal.',
    },
  },
}

const profile: ScalingReport = {
  report_id: 'c'.repeat(64),
  kind: 'hardware-profile',
  scope: 'host-measurement',
  claim_boundary: 'Measurement of the identified configuration on this host only.',
  inputs: {},
  warnings: ['MFU is unknown: no operator-supplied peak FLOP/s.'],
  measured: { tokens_per_second: 2570.7, step_seconds_mean: 0.05, mfu: null, mfu_status: 'unknown' },
  summary: listing.reports[0],
  results: {},
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ScalingPage', () => {
  it('separates local rungs from projection-only rungs and offers no launch control', async () => {
    vi.spyOn(api, 'ladder').mockResolvedValue(ladder)
    vi.spyOn(api, 'scalingReports').mockResolvedValue(listing)
    const onNew = vi.fn()
    render(<ScalingPage onNew={onNew} tokenizedId={'9'.repeat(64)} />)

    const rungs = await screen.findByRole('table', { name: 'Ladder rungs' })
    expect(within(rungs).getByText('local')).toBeInTheDocument()
    expect(within(rungs).getByText('projection only')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /launch|provision|approve/i })).toBeNull()

    await userEvent.click(screen.getByRole('button', { name: 'Sweep' }))
    expect(onNew).toHaveBeenCalledWith('scale_sweep', { training: JSON.stringify({ tokenized_id: '9'.repeat(64) }) })
    const reports = await screen.findByRole('table', { name: 'Stored scaling reports' })
    expect(within(reports).getByText(/MFU unknown/)).toBeInTheDocument()
    expect(within(reports).getByText(/approval required/)).toBeInTheDocument()
  })

  it('shows a projection with its unknown inputs and the approval requirement prominently', async () => {
    vi.spyOn(api, 'ladder').mockResolvedValue(ladder)
    vi.spyOn(api, 'scalingReports').mockResolvedValue(listing)
    vi.spyOn(api, 'scalingReport').mockResolvedValue([projection])
    render(<ScalingPage onNew={vi.fn()} tokenizedId={null} />)

    const reports = await screen.findByRole('table', { name: 'Stored scaling reports' })
    await userEvent.click(within(reports).getAllByRole('button', { name: 'View' })[1])
    const detail = await screen.findByRole('region', { name: `Scaling report ${'e'.repeat(8)}` })
    expect(within(detail).getByRole('alert')).toHaveTextContent('Approval required')
    expect(within(detail).getByRole('alert')).toHaveTextContent('not authorization')
    expect(within(detail).getByRole('note')).toHaveTextContent('Unknown inputs (3)')
    expect(within(detail).getByRole('note')).toHaveTextContent('hardware.peak_flops_per_device')
    const estimates = within(detail).getByRole('table', { name: 'Projection estimates' })
    expect(within(estimates).getAllByText('unknown').length).toBeGreaterThan(0)
    expect(within(estimates).getByText('20.00B tokens')).toBeInTheDocument()
    expect(within(detail).queryByRole('button')).toBeNull()
  })

  it('reports MFU as unknown when no device peak was supplied', async () => {
    vi.spyOn(api, 'ladder').mockResolvedValue(ladder)
    vi.spyOn(api, 'scalingReports').mockResolvedValue(listing)
    vi.spyOn(api, 'scalingReport').mockResolvedValue([profile])
    render(<ScalingPage onNew={vi.fn()} tokenizedId={null} />)

    const reports = await screen.findByRole('table', { name: 'Stored scaling reports' })
    await userEvent.click(within(reports).getAllByRole('button', { name: 'View' })[0])
    const table = await screen.findByRole('table', { name: 'Host profile' })
    expect(within(table).getByText(/unknown — no operator-supplied device peak/)).toBeInTheDocument()
    expect(within(table).getByText('2.6K')).toBeInTheDocument()
  })
})
