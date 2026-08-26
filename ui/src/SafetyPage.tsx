import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { EmptyState, StatusPill, formatDate, shortId, titleCase, useDialogBehavior } from './shared'
import type {
  ActionRegistry,
  Covenant,
  Escalation,
  EscalationState,
  Identity,
} from './types'

const escalationPollMilliseconds = 4_000
const stateFilters: (EscalationState | 'all')[] = [
  'open',
  'approved',
  'refused',
  'consumed',
  'expired',
  'all',
]

/**
 * The covenant page renders what the resolver is executing right now, read
 * back from the API. Nothing here is restated in the front end: a paraphrase
 * would drift from the tracked text, which is the failure the hash exists to
 * make visible.
 */
export function CovenantCard({ covenant }: { covenant: Covenant }) {
  const { ratification } = covenant
  const inForce = covenant.status === 'ratified' && ratification.matches_covenant
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>
            Covenant v{covenant.covenant_version} <StatusPill status={covenant.status} />
          </h2>
          <p>
            Effective {covenant.effective_date} for roadmap {covenant.roadmap_version}. An
            assessment below this confidence floor ({covenant.escalation_confidence_floor}) reaches
            a human instead of proceeding.
          </p>
        </div>
      </div>
      <dl className="fact-list">
        <div>
          <dt>Canonical hash</dt>
          <dd>
            <code className="hash">{covenant.sha256}</code>
          </dd>
        </div>
        <div>
          <dt>Ratification</dt>
          <dd>
            {!ratification.present
              ? 'No ratification manifest is tracked.'
              : inForce
                ? `Ratified ${ratification.ratified_on} by ${ratification.reviewer_role}.`
                : `Prepared by ${ratification.reviewer_role}, not yet in force: the covenant is still a draft.`}
          </dd>
        </div>
        {ratification.evidence.length > 0 && (
          <div>
            <dt>Evidence</dt>
            <dd className="evidence-list">
              {ratification.evidence.map((item) => (
                <code key={item}>{item}</code>
              ))}
            </dd>
          </div>
        )}
      </dl>
    </section>
  )
}

export function PrincipleList({ covenant }: { covenant: Covenant }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Principles, in strict priority order</h2>
          <p>
            Lower priority outranks higher. A conflict is permitted only when the action protects a
            strictly higher-ranked principle.
          </p>
        </div>
      </div>
      <ol className="principle-list">
        {covenant.principles.map((principle) => (
          <li key={principle.key}>
            <div className="principle-head">
              <span className="principle-rank" aria-hidden="true">
                {principle.priority}
              </span>
              <div>
                <h3>
                  <span className="sr-only">Principle {principle.priority}: </span>
                  {principle.title}
                </h3>
                <small>{principle.origin}</small>
              </div>
            </div>
            <p className="principle-law">{principle.law}</p>
            <dl className="obligation-list">
              {principle.obligations.map((obligation) => (
                <div key={obligation.key}>
                  <dt>
                    <code>{obligation.key}</code>
                  </dt>
                  <dd>{obligation.statement}</dd>
                </div>
              ))}
            </dl>
          </li>
        ))}
      </ol>
    </section>
  )
}

export function EnforcementCard({ covenant }: { covenant: Covenant }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Where this is enforced</h2>
          <p>The checkpoints that execute the covenant, and what each one decides.</p>
        </div>
      </div>
      <dl className="fact-list">
        {covenant.enforcement_checkpoints.map((checkpoint) => (
          <div key={checkpoint.name}>
            <dt>{checkpoint.name}</dt>
            <dd className="checkpoint-copy">{checkpoint.description}</dd>
          </div>
        ))}
      </dl>
      <div className="panel-footer">
        <p className="muted-line">
          <strong>Limitations.</strong> {covenant.limitations}
        </p>
      </div>
    </section>
  )
}

export function ActionRegistryTable({ registry }: { registry: ActionRegistry }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <h2>{registry.count} declared actions</h2>
          <p>{registry.undeclared_behavior}</p>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Action</th>
              <th>Standing</th>
              <th>Surface</th>
              <th>Summary</th>
            </tr>
          </thead>
          <tbody>
            {registry.items.map((action) => (
              <tr key={action.action_id}>
                <td className="job-name">
                  <button
                    type="button"
                    className="job-row-trigger"
                    aria-expanded={expanded === action.action_id}
                    onClick={() =>
                      setExpanded(expanded === action.action_id ? null : action.action_id)
                    }
                  >
                    <strong>{action.action_id}</strong>
                    <code>{action.job_type ? 'job type' : 'command only'}</code>
                  </button>
                  {expanded === action.action_id && (
                    <dl className="assessment-list">
                      {action.assessments.map((assessment) => (
                        <div key={assessment.priority}>
                          <dt>
                            Principle {assessment.priority} · {assessment.status} ·{' '}
                            {assessment.confidence}
                          </dt>
                          <dd>{assessment.rationale}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </td>
                <td>
                  <StatusPill status={action.outcome} />
                </td>
                <td>
                  {action.cli_commands.length === 0 ? (
                    <span className="muted-line">API only</span>
                  ) : (
                    action.cli_commands.map((command) => <code key={command}>{command}</code>)
                  )}
                </td>
                <td>{action.summary}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export function EscalationDetail({
  escalation,
  identity,
  onClose,
  onDecided,
}: {
  escalation: Escalation
  identity: Identity
  onClose: () => void
  onDecided: () => void
}) {
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLDivElement>(onClose)
  const [rationale, setRationale] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const decidable = escalation.state === 'open' && identity.role === 'owner'

  const decide = async (approve: boolean) => {
    setBusy(true)
    setError('')
    try {
      await api.decideEscalation(escalation.id, approve, rationale)
      onDecided()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not record the decision')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="escalation-title"
        tabIndex={-1}
        onKeyDown={onDialogKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-head">
          <div>
            <h2 id="escalation-title">{escalation.action_id}</h2>
            <p>
              Raised by {escalation.actor_role} {shortId(escalation.actor_id)} on{' '}
              {formatDate(escalation.created_at)}
            </p>
            <code>{shortId(escalation.id)}</code>
          </div>
          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="Close"
            data-dialog-initial-focus
          >
            ×
          </button>
        </div>
        <div className="drawer-status">
          <StatusPill status={escalation.state} />
          <span>
            {escalation.binding_priority === null
              ? 'no binding principle'
              : `principle ${escalation.binding_priority}`}
          </span>
        </div>
        <ul className="check-list">
          {escalation.reasons.map((reason) => (
            <li key={reason}>
              <span aria-hidden="true">·</span>
              {reason}
            </li>
          ))}
        </ul>
        <dl className="fact-list">
          <div>
            <dt>Request digest</dt>
            <dd>
              <code className="hash">{escalation.request_digest}</code>
            </dd>
          </div>
          <div>
            <dt>Deciding covenant</dt>
            <dd>
              <code className="hash">{escalation.covenant_sha256}</code>
            </dd>
          </div>
          <div>
            <dt>Approval window</dt>
            <dd>{formatDate(escalation.approval_expires_at)}</dd>
          </div>
          {escalation.rationale && (
            <div>
              <dt>Recorded rationale</dt>
              <dd>{escalation.rationale}</dd>
            </div>
          )}
          {escalation.consumed_job_id && (
            <div>
              <dt>Authorized job</dt>
              <dd>
                <code>{shortId(escalation.consumed_job_id)}</code>
              </dd>
            </div>
          )}
        </dl>
        <p className="muted-line drawer-note">
          The submitted specification is deliberately not recorded. An approval authorizes this
          exact request digest, once, and the requester must submit it again.
        </p>
        {decidable ? (
          <>
            <label className="form-field">
              <span>Decision rationale</span>
              <textarea
                rows={3}
                aria-label="Decision rationale"
                value={rationale}
                onChange={(event) => setRationale(event.target.value)}
                aria-describedby="escalation-rationale-help"
              />
              <small id="escalation-rationale-help">
                Recorded in safety truth alongside the decision. Required.
              </small>
            </label>
            {error && <div className="error-banner">{error}</div>}
            <div className="drawer-actions">
              <button
                type="button"
                className="primary-button"
                disabled={busy || rationale.trim() === ''}
                onClick={() => void decide(true)}
              >
                Approve once
              </button>
              <button
                type="button"
                className="danger-button"
                disabled={busy || rationale.trim() === ''}
                onClick={() => void decide(false)}
              >
                Refuse
              </button>
            </div>
          </>
        ) : (
          <p className="muted-line drawer-note">
            {escalation.state !== 'open'
              ? 'This escalation has already been decided.'
              : 'Only the project owner may decide an escalation.'}
          </p>
        )}
      </div>
    </div>
  )
}

export function SafetyPage({ identity }: { identity: Identity }) {
  const [covenant, setCovenant] = useState<Covenant | null>(null)
  const [registry, setRegistry] = useState<ActionRegistry | null>(null)
  const [escalations, setEscalations] = useState<Escalation[]>([])
  const [filter, setFilter] = useState<EscalationState | 'all'>('open')
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let disposed = false
    Promise.all([api.covenant(), api.actions()])
      .then(([nextCovenant, nextRegistry]) => {
        if (disposed) return
        setCovenant(nextCovenant)
        setRegistry(nextRegistry)
      })
      .catch((caught: unknown) => {
        if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not load covenant')
      })
    return () => {
      disposed = true
    }
  }, [])

  const refresh = useCallback(async () => {
    setEscalations(await api.escalations(filter === 'all' ? undefined : filter))
  }, [filter])

  useEffect(() => {
    let disposed = false
    let timer: number | undefined
    // Self-scheduling rather than an interval: a slow response can never stack
    // a second request on top of the one still in flight.
    const poll = async () => {
      try {
        await refresh()
      } catch {
        /* the global error banner already covers a dead API */
      } finally {
        if (!disposed) timer = window.setTimeout(() => void poll(), escalationPollMilliseconds)
      }
    }
    void poll()
    return () => {
      disposed = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [refresh])

  const active = escalations.find((item) => item.id === selected) ?? null
  const openCount = escalations.filter((item) => item.state === 'open').length

  return (
    <div className="page-stack">
      {error && (
        <div className="global-error" role="alert">
          <span>{error}</span>
        </div>
      )}
      {covenant && <CovenantCard covenant={covenant} />}
      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Escalation queue</h2>
            <p>
              Work the covenant would not decide. No job was created, and no submitted
              specification was stored.
            </p>
          </div>
          <label className="form-field inline-field">
            <span className="sr-only">Filter escalations by state</span>
            <select
              aria-label="Filter escalations by state"
              value={filter}
              onChange={(event) => setFilter(event.target.value as EscalationState | 'all')}
            >
              {stateFilters.map((item) => (
                <option key={item} value={item}>
                  {titleCase(item)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="sr-only" role="status">
          {openCount} escalation{openCount === 1 ? '' : 's'} awaiting review.
        </p>
        {escalations.length === 0 ? (
          <EmptyState>Nothing has been escalated under this filter.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Action</th>
                  <th>State</th>
                  <th>Principle</th>
                  <th>Raised</th>
                </tr>
              </thead>
              <tbody>
                {escalations.map((item) => (
                  <tr className="clickable-row" key={item.id} onClick={() => setSelected(item.id)}>
                    <td className="job-name">
                      <button
                        type="button"
                        className="job-row-trigger"
                        aria-label={`Review ${item.action_id} escalation ${shortId(item.id)}`}
                      >
                        <strong>{item.action_id}</strong>
                        <code>{shortId(item.id)}</code>
                      </button>
                    </td>
                    <td>
                      <StatusPill status={item.state} />
                    </td>
                    <td className="numeric-cell">{item.binding_priority ?? '—'}</td>
                    <td className="numeric-cell">{formatDate(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {covenant && <PrincipleList covenant={covenant} />}
      {registry && <ActionRegistryTable registry={registry} />}
      {covenant && <EnforcementCard covenant={covenant} />}
      {active && (
        <EscalationDetail
          escalation={active}
          identity={identity}
          onClose={() => setSelected(null)}
          onDecided={() => {
            setSelected(null)
            void refresh()
          }}
        />
      )}
    </div>
  )
}
