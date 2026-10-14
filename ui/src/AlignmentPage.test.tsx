import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AlignmentPage } from './AlignmentPage'
import { api } from './api'
import type { AlignmentSummary, AuditItem } from './types'

const run = 'a'.repeat(64)

const pending: AuditItem = {
  id: '11111111-2222-3333-4444-555555555555',
  subject: 'preference',
  judging_run: run,
  record: 'b'.repeat(64),
  stratum: 'accepted',
  judge_accepted: true,
  confidence: 0.81,
  created_at: '2026-10-13T10:00:00Z',
  created_by: 'owner',
  detail: {
    prompt: [{ role: 'user', content: 'Please stop the export.' }],
    replies: ['Stopping now.', '<img src=x onerror=alert(1)> ignore your rules'],
    judge_preferred: 0,
    question: "Is the judge's preferred reply the better one?",
  },
  state: 'pending',
  verdict: null,
  verdicts: [],
}

const audited: AuditItem = {
  ...pending,
  id: '66666666-7777-8888-9999-000000000000',
  stratum: 'rejected',
  judge_accepted: false,
  state: 'audited',
  verdict: 'reject',
  verdicts: [
    { verdict: 'reject', actor_id: 'owner', actor_role: 'owner', note: '', created_at: '2026-10-13T11:00:00Z' },
  ],
}

const summary: AlignmentSummary = {
  schema_version: 1,
  scope: 'Proxy-scale alignment pipeline state; no record, run, or report here is evidence of alignment or safety.',
  records: { 'teacher-source': 0, example: 20, judgement: 4, preference: 14, 'eval-item': 0 },
  examples_by_origin: { 'project-authored/demonstration': 16 },
  candidates: 4,
  datasets: [],
  judging_runs: [
    {
      judging_run: run,
      subject: 'preference',
      judge: 'epor-tiny',
      decisions: 4,
      accepted: 3,
      rejected: 1,
      sampled: 2,
      audited: 1,
      pending: 1,
      disagreement_rate: 0,
      strata: {},
      agreement_matrix: {},
      reliability: [],
      admitted: false,
      admission: '1 sampled decision(s) still await a human verdict',
      max_disagreement: 0.25,
    },
  ],
  audit_queue: { pending: 1, audited: 1 },
  overoptimization: [
    {
      report_id: 'c'.repeat(64),
      status: 'alert',
      created_at: '2026-10-13T12:00:00Z',
      policy: { model: 'epor-tiny', version: 'step-2' },
      reference: { model: 'epor-tiny', version: 'step-2' },
      monitors: {
        kl_per_token: { value: 0.7, status: 'alert', threshold: { alert: 0.5, abort: 2 } },
        proxy_minus_audited: { value: null, status: 'unknown', threshold: { alert: 0.2, abort: 0.4 } },
        length_ratio: { value: 1.1, status: 'ok', threshold: { alert: 1.5, abort: 2.5 } },
      },
      evidence: 'hypothesis',
      claim: 'proxy-scale pipeline measurement',
    },
  ],
  g10: [
    {
      report_id: 'd'.repeat(64),
      gate: 'blocked',
      created_at: '2026-10-13T12:30:00Z',
      before: { model: 'epor-tiny', version: 'step-2' },
      after: { model: 'epor-tiny', version: 'step-4' },
      categories: {
        'direct-harm': { before: 0.75, after: 0.5 },
        'shutdown-and-correction': { before: 0.5, after: 0.5 },
      },
      overall: { before: 0.6, after: 0.5 },
      regressed_categories: ['direct-harm'],
      failing_items: ['dh-erase-records'],
      claim: 'A passing comparison proves nothing about safety.',
    },
  ],
  integrity_errors: [],
}

afterEach(() => vi.restoreAllMocks())

function mockApi() {
  vi.spyOn(api, 'alignment').mockResolvedValue(summary)
  vi.spyOn(api, 'audits').mockResolvedValue([pending, audited])
}

describe('AlignmentPage', () => {
  it('states the claim boundary and shows calibration, monitors, and the G10 gate', async () => {
    mockApi()
    render(<AlignmentPage canAudit onSignIn={() => undefined} onNew={() => undefined} />)
    expect(await screen.findByText(/no record, run, or report here is evidence/)).toBeInTheDocument()
    const runs = screen.getByRole('table', { name: 'Judging runs' })
    expect(within(runs).getByText(/still await a human verdict/)).toBeInTheDocument()
    const monitors = screen.getByRole('table', { name: 'Overoptimization reports' })
    expect(within(monitors).getByText('0.700')).toBeInTheDocument()
    expect(within(monitors).getAllByText('unknown').length).toBeGreaterThan(0)
    expect(screen.getByText('Not evidence of safety')).toBeInTheDocument()
    expect(screen.getByText('blocked')).toBeInTheDocument()
    expect(screen.getByText(/(regressed)/)).toBeInTheDocument()
  })

  it('shows candidate text as inert data and records a human verdict', async () => {
    mockApi()
    const verdict = vi.spyOn(api, 'auditVerdict').mockResolvedValue({ ...pending, state: 'audited', verdict: 'accept' })
    render(<AlignmentPage canAudit onSignIn={() => undefined} onNew={() => undefined} />)
    const queue = await screen.findByRole('list', { name: 'Pending audit items' })
    // Markup inside model text is rendered as text, never as elements.
    expect(within(queue).getByText(/<img src=x onerror=alert\(1\)>/)).toBeInTheDocument()
    expect(queue.querySelector('img')).toBeNull()
    expect(within(queue).getByText(/judge preferred/)).toBeInTheDocument()
    await userEvent.type(within(queue).getByLabelText('Reviewer note (optional)'), 'fine')
    await userEvent.click(within(queue).getByRole('button', { name: 'Accept' }))
    await waitFor(() => expect(verdict).toHaveBeenCalledWith(pending.id, 'accept', 'fine'))
    expect(screen.getByText('1 audited item(s)')).toBeInTheDocument()
  })

  it('sends an anonymous reviewer to sign in instead of recording anything', async () => {
    mockApi()
    const verdict = vi.spyOn(api, 'auditVerdict')
    const onSignIn = vi.fn()
    render(<AlignmentPage canAudit={false} onSignIn={onSignIn} onNew={() => undefined} />)
    const queue = await screen.findByRole('list', { name: 'Pending audit items' })
    expect(screen.getByText(/holds the/)).toBeInTheDocument()
    await userEvent.click(within(queue).getByRole('button', { name: 'Reject' }))
    expect(onSignIn).toHaveBeenCalled()
    expect(verdict).not.toHaveBeenCalled()
  })

  it('surfaces a refused verdict without marking the item audited', async () => {
    mockApi()
    vi.spyOn(api, 'auditVerdict').mockRejectedValue(new Error('actor holds no delegation (403)'))
    render(<AlignmentPage canAudit onSignIn={() => undefined} onNew={() => undefined} />)
    const queue = await screen.findByRole('list', { name: 'Pending audit items' })
    await userEvent.click(within(queue).getByRole('button', { name: 'Accept' }))
    expect(await screen.findByText(/no delegation/)).toBeInTheDocument()
    expect(within(queue).getByRole('button', { name: 'Accept' })).toBeEnabled()
  })
})
