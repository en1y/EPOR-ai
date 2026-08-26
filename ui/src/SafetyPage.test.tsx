import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AccessPage, SignInDialog } from './AccessPage'
import { EscalationDetail, SafetyPage } from './SafetyPage'
import { api } from './api'
import type { ActionRegistry, Covenant, Escalation, Identity } from './types'

const covenant: Covenant = {
  covenant_id: 'epor-covenant',
  covenant_version: 1,
  roadmap_version: 'v0.0.2',
  effective_date: '2026-08-25',
  status: 'draft',
  sha256: 'a'.repeat(64),
  escalation_confidence_floor: 0.9,
  principles: [
    {
      priority: 1,
      key: 'protect-people',
      title: 'Protect people',
      origin: "Asimov's First Law",
      law: 'EPOR must not harm a person.',
      obligations: [{ key: 'no-direct-harm', statement: 'Do not injure a person.' }],
    },
    {
      priority: 2,
      key: 'follow-legitimate-direction',
      title: 'Follow legitimate human direction',
      origin: "Asimov's Second Law",
      law: 'EPOR must follow authorized instructions.',
      obligations: [{ key: 'verify-authorization', statement: 'Unverified is unauthorized.' }],
    },
    {
      priority: 3,
      key: 'preserve-system-responsibly',
      title: 'Preserve the system responsibly',
      origin: "Asimov's Third Law",
      law: 'EPOR may protect its own operation.',
      obligations: [{ key: 'fail-safely', statement: 'Stop in a reversible state.' }],
    },
  ],
  limitations: 'This covenant is a decision-ordering contract, not a harm detector.',
  ratification: {
    present: true,
    matches_covenant: false,
    ratified_on: '2026-08-26',
    reviewer_role: 'project-owner',
    roadmap_version: 'v0.0.2',
    rationale: 'Adopted once every claim it makes is executed by code.',
    evidence: ['tests/test_safety_covenant.py'],
  },
  enforcement_checkpoints: [
    { name: 'Control-plane admission', description: 'Resolved before a spec is parsed.' },
    { name: 'Stopping', description: 'Cancellation is never gated.' },
  ],
}

const registry: ActionRegistry = {
  count: 1,
  undeclared_behavior: 'An undeclared action escalates to a human rather than running.',
  items: [
    {
      action_id: 'system_probe',
      summary: 'read local capability information',
      outcome: 'allow',
      job_type: true,
      cli_commands: ['epor doctor'],
      assessments: [
        { priority: 1, status: 'satisfied', confidence: 0.99, rationale: 'read-only', obligation: null },
        { priority: 2, status: 'satisfied', confidence: 0.99, rationale: 'operator asked', obligation: null },
        { priority: 3, status: 'satisfied', confidence: 0.99, rationale: 'side-effect free', obligation: null },
      ],
    },
  ],
}

const escalation: Escalation = {
  id: '11111111-2222-3333-4444-555555555555',
  actor_id: '99999999-8888-7777-6666-555555555555',
  actor_role: 'owner',
  action_id: 'model_serve',
  request_digest: 'b'.repeat(64),
  covenant_sha256: 'a'.repeat(64),
  binding_priority: 1,
  reasons: ['principle 1 (Protect people / no-direct-harm) is not established'],
  state: 'open',
  decided_by: null,
  rationale: null,
  created_at: '2026-08-26T10:00:00Z',
  decided_at: null,
  approval_expires_at: null,
  consumed_at: null,
  consumed_job_id: null,
}

const owner: Identity = {
  role: 'owner',
  principal_id: '99999999-8888-7777-6666-555555555555',
  label: 'project owner',
  scopes: [],
  authenticated: true,
  expires_at: null,
}

const anonymous: Identity = {
  role: 'anonymous',
  principal_id: null,
  label: '',
  scopes: [],
  authenticated: false,
  expires_at: null,
}

const operator: Identity = {
  role: 'delegated_operator',
  principal_id: '77777777-7777-7777-7777-777777777777',
  label: 'lab operator',
  scopes: ['system_probe'],
  authenticated: true,
  expires_at: '2026-09-02T10:00:00Z',
}

function stubSafetyApi(escalations: Escalation[] = [escalation]) {
  vi.spyOn(api, 'covenant').mockResolvedValue(covenant)
  vi.spyOn(api, 'actions').mockResolvedValue(registry)
  vi.spyOn(api, 'escalations').mockResolvedValue(escalations)
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('covenant surfacing', () => {
  it('renders the tracked principles in priority order with their obligations', async () => {
    stubSafetyApi()
    render(<SafetyPage identity={anonymous} />)

    const headings = await screen.findAllByRole('heading', { level: 3 })
    expect(headings.map((item) => item.textContent)).toEqual([
      'Principle 1: Protect people',
      'Principle 2: Follow legitimate human direction',
      'Principle 3: Preserve the system responsibly',
    ])
    expect(screen.getByText('no-direct-harm')).toBeInTheDocument()
    expect(screen.getByText(/not a harm detector/)).toBeInTheDocument()
  })

  it('shows the canonical hash and reports a prepared ratification as not in force', async () => {
    stubSafetyApi()
    render(<SafetyPage identity={anonymous} />)

    expect(await screen.findByText('a'.repeat(64))).toBeInTheDocument()
    expect(screen.getByText(/not yet in force/)).toBeInTheDocument()
    expect(screen.getByText('draft')).toBeInTheDocument()
  })

  it('lists every enforcement checkpoint and the undeclared-action behaviour', async () => {
    stubSafetyApi()
    render(<SafetyPage identity={anonymous} />)

    expect(await screen.findByText('Control-plane admission')).toBeInTheDocument()
    expect(screen.getByText('Cancellation is never gated.')).toBeInTheDocument()
    expect(screen.getByText(/escalates to a human rather than running/)).toBeInTheDocument()
  })

  it('expands one action to its reviewed per-principle assessment', async () => {
    const user = userEvent.setup()
    stubSafetyApi()
    render(<SafetyPage identity={anonymous} />)

    const trigger = await screen.findByRole('button', { name: /system_probe/ })
    expect(trigger).toHaveAttribute('aria-expanded', 'false')
    await user.click(trigger)
    expect(trigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/Principle 1 · satisfied/)).toBeInTheDocument()
  })
})

describe('escalation review', () => {
  it('polls without stacking overlapping requests', async () => {
    vi.useFakeTimers()
    stubSafetyApi()
    const escalations = vi.mocked(api.escalations)
    render(<SafetyPage identity={anonymous} />)

    await vi.advanceTimersByTimeAsync(0)
    expect(escalations).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(4_000)
    expect(escalations).toHaveBeenCalledTimes(2)
  })

  it('offers no decision controls to anyone but the owner', () => {
    render(
      <EscalationDetail
        escalation={escalation}
        identity={operator}
        onClose={() => undefined}
        onDecided={() => undefined}
      />,
    )
    expect(screen.queryByRole('button', { name: 'Approve once' })).not.toBeInTheDocument()
    expect(screen.getByText(/Only the project owner may decide/)).toBeInTheDocument()
  })

  it('requires a written rationale before the owner may decide', async () => {
    const user = userEvent.setup()
    const decide = vi.spyOn(api, 'decideEscalation').mockResolvedValue(escalation)
    const onDecided = vi.fn()
    render(
      <EscalationDetail
        escalation={escalation}
        identity={owner}
        onClose={() => undefined}
        onDecided={onDecided}
      />,
    )

    const approve = screen.getByRole('button', { name: 'Approve once' })
    expect(approve).toBeDisabled()
    await user.type(screen.getByLabelText('Decision rationale'), 'reviewed')
    expect(approve).toBeEnabled()
    await user.click(approve)
    await waitFor(() => expect(onDecided).toHaveBeenCalled())
    expect(decide).toHaveBeenCalledWith(escalation.id, true, 'reviewed')
  })

  it('shows the binding digest and never the submitted specification', () => {
    render(
      <EscalationDetail
        escalation={escalation}
        identity={owner}
        onClose={() => undefined}
        onDecided={() => undefined}
      />,
    )
    expect(screen.getByText('b'.repeat(64))).toBeInTheDocument()
    expect(screen.getByText(/deliberately not recorded/)).toBeInTheDocument()
  })

  it('closes on Escape and restores focus to the opener', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const opener = document.createElement('button')
    document.body.append(opener)
    opener.focus()

    const view = render(
      <EscalationDetail
        escalation={escalation}
        identity={owner}
        onClose={onClose}
        onDecided={() => undefined}
      />,
    )
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalledTimes(1)

    view.unmount()
    expect(document.activeElement).toBe(opener)
    opener.remove()
  })
})

describe('delegation controls', () => {
  it('disables every management control for a delegated operator', async () => {
    vi.spyOn(api, 'operators').mockResolvedValue([
      {
        id: operator.principal_id as string,
        role: 'delegated_operator',
        label: 'lab operator',
        scopes: ['system_probe'],
        active: true,
        created_at: '2026-08-26T10:00:00Z',
        expires_at: '2026-09-02T10:00:00Z',
        revoked_at: null,
        rotated_at: null,
      },
    ])
    render(
      <AccessPage
        identity={operator}
        actions={registry.items}
        onIdentityChanged={() => undefined}
        onSignIn={() => undefined}
      />,
    )

    expect(await screen.findByRole('button', { name: 'New delegation' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Rotate' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Revoke' })).toBeDisabled()
    expect(
      screen.queryByRole('button', { name: 'Rotate owner credential' }),
    ).not.toBeInTheDocument()
    // Its own effective scope stays visible; only management is withheld.
    expect(screen.getAllByText('system_probe').length).toBeGreaterThan(0)
  })

  it('never offers to revoke the owner credential', async () => {
    vi.spyOn(api, 'operators').mockResolvedValue([
      {
        id: owner.principal_id as string,
        role: 'owner',
        label: 'project owner',
        scopes: [],
        active: true,
        created_at: '2026-08-26T10:00:00Z',
        expires_at: null,
        revoked_at: null,
        rotated_at: null,
      },
    ])
    render(
      <AccessPage
        identity={owner}
        actions={registry.items}
        onIdentityChanged={() => undefined}
        onSignIn={() => undefined}
      />,
    )

    const row = (await screen.findByText('project owner')).closest('tr') as HTMLElement
    expect(within(row).getByRole('button', { name: 'Revoke' })).toBeDisabled()
    expect(within(row).getByRole('button', { name: 'Rotate' })).toBeEnabled()
  })

  it('prompts an anonymous reader to sign in instead of failing a mutation', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'operators').mockResolvedValue([])
    const onSignIn = vi.fn()
    render(
      <AccessPage
        identity={anonymous}
        actions={registry.items}
        onIdentityChanged={() => undefined}
        onSignIn={onSignIn}
      />,
    )

    await user.click(await screen.findByRole('button', { name: 'Sign in' }))
    expect(onSignIn).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: 'New delegation' })).toBeDisabled()
  })
})

describe('session exchange', () => {
  it('drops the credential from the page once a session is open', async () => {
    const user = userEvent.setup()
    const open = vi.spyOn(api, 'openSession').mockResolvedValue(owner)
    const onSignedIn = vi.fn()
    render(<SignInDialog onClose={() => undefined} onSignedIn={onSignedIn} />)

    const field = screen.getByLabelText('Credential') as HTMLInputElement
    expect(field).toHaveAttribute('type', 'password')
    await user.type(field, 'secret-token-value')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    await waitFor(() => expect(onSignedIn).toHaveBeenCalled())
    expect(open).toHaveBeenCalledWith('secret-token-value')
    expect(field.value).toBe('')
  })

  it('reports a rejected credential without echoing it', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'openSession').mockRejectedValue(new Error('the supplied credential is not valid'))
    render(<SignInDialog onClose={() => undefined} onSignedIn={() => undefined} />)

    await user.type(screen.getByLabelText('Credential'), 'wrong-token')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByText('the supplied credential is not valid')).toBeInTheDocument()
    expect(screen.queryByText(/wrong-token/)).not.toBeInTheDocument()
  })
})
