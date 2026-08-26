import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { EmptyState, StatusPill, formatDate, titleCase, useDialogBehavior } from './shared'
import type { DeclaredAction, Identity, IssuedCredential, Operator } from './types'

const defaultDelegationDays = 7
const maximumDelegationDays = 30

function isoDaysFromNow(days: number): string {
  return new Date(Date.now() + days * 86_400_000).toISOString()
}

/**
 * A credential is displayed exactly once, here, immediately after it is
 * minted. The server keeps only its digest, so there is no second chance to
 * read it and no endpoint that could hand it back.
 */
export function IssuedCredentialDialog({
  issued,
  onClose,
}: {
  issued: IssuedCredential
  onClose: () => void
}) {
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLDivElement>(onClose)
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className="job-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="issued-title"
        tabIndex={-1}
        onKeyDown={onDialogKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-head">
          <div>
            <h2 id="issued-title">Credential for {issued.operator.label}</h2>
            <p>Shown once. Only a SHA-256 digest is stored, so this cannot be retrieved again.</p>
          </div>
        </div>
        <div className="form-field">
          <span>Token</span>
          <input readOnly value={issued.token} aria-label="Issued credential" />
        </div>
        <div className="modal-actions">
          <button
            type="button"
            className="primary-button"
            data-dialog-initial-focus
            onClick={onClose}
          >
            I have copied it
          </button>
        </div>
      </div>
    </div>
  )
}

export function DelegationForm({
  actions,
  onClose,
  onIssued,
}: {
  actions: DeclaredAction[]
  onClose: () => void
  onIssued: (issued: IssuedCredential) => void
}) {
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLFormElement>(onClose)
  const [label, setLabel] = useState('')
  const [days, setDays] = useState(defaultDelegationDays)
  const [scopes, setScopes] = useState<string[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      onIssued(await api.createOperator(label, scopes, isoDaysFromNow(days)))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not create the delegation')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <form
        ref={dialogRef}
        className="job-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delegation-title"
        tabIndex={-1}
        onSubmit={(event) => void submit(event)}
        onKeyDown={onDialogKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-head">
          <div>
            <h2 id="delegation-title">Delegate to an operator</h2>
            <p>
              A delegation names declared actions explicitly and always expires. There is no
              non-expiring operator credential.
            </p>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <label className="form-field">
          <span>Label</span>
          <input
            required
            aria-label="Label"
            data-dialog-initial-focus
            value={label}
            maxLength={120}
            onChange={(event) => setLabel(event.target.value)}
          />
        </label>
        <label className="form-field">
          <span>Expires in days</span>
          <input
            required
            type="number"
            aria-label="Expires in days"
            min={1}
            max={maximumDelegationDays}
            value={days}
            onChange={(event) => setDays(Number(event.target.value))}
          />
          <small>At most {maximumDelegationDays} days.</small>
        </label>
        <fieldset className="scope-field">
          <legend>Delegated actions</legend>
          {actions.map((action) => (
            <label className="checkbox-field" key={action.action_id}>
              <input
                type="checkbox"
                checked={scopes.includes(action.action_id)}
                onChange={(event) =>
                  setScopes((current) =>
                    event.target.checked
                      ? [...current, action.action_id]
                      : current.filter((item) => item !== action.action_id),
                  )
                }
              />
              <span>
                {action.action_id}
                <small>{action.summary}</small>
              </span>
            </label>
          ))}
        </fieldset>
        {error && <div className="error-banner">{error}</div>}
        <div className="modal-actions">
          <button type="button" className="ghost-button" onClick={onClose}>
            Dismiss
          </button>
          <button
            className="primary-button"
            type="submit"
            disabled={busy || scopes.length === 0 || label.trim() === ''}
          >
            {busy ? 'Creating…' : 'Create delegation'}
          </button>
        </div>
      </form>
    </div>
  )
}

export function AccessPage({
  identity,
  actions,
  onIdentityChanged,
  onSignIn,
}: {
  identity: Identity
  actions: DeclaredAction[]
  onIdentityChanged: () => void
  onSignIn: () => void
}) {
  const [operators, setOperators] = useState<Operator[]>([])
  const [issued, setIssued] = useState<IssuedCredential | null>(null)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const isOwner = identity.role === 'owner'

  const refresh = useCallback(async () => {
    try {
      setOperators(await api.operators())
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not load principals')
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const act = async (run: () => Promise<unknown>) => {
    setError('')
    try {
      const result = await run()
      if (result && typeof result === 'object' && 'token' in result) {
        setIssued(result as IssuedCredential)
      }
      await refresh()
      onIdentityChanged()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The request was not permitted')
    }
  }

  return (
    <div className="page-stack">
      {error && (
        <div className="global-error" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError('')}>
            Dismiss
          </button>
        </div>
      )}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>
              You are {identity.authenticated ? identity.label || titleCase(identity.role) : 'anonymous'}{' '}
              <StatusPill status={identity.role} />
            </h2>
            <p>
              {identity.authenticated
                ? 'Reading is public. Starting work requires a delegation covering that action.'
                : 'Every read-only page is browsable without signing in. Mutations need a credential.'}
            </p>
          </div>
          {identity.authenticated ? (
            isOwner && (
              <button
                type="button"
                className="ghost-button"
                onClick={() => void act(() => api.rotateOwner())}
              >
                Rotate owner credential
              </button>
            )
          ) : (
            <button type="button" className="primary-button" onClick={onSignIn}>
              Sign in
            </button>
          )}
        </div>
        <dl className="fact-list">
          <div>
            <dt>Effective scopes</dt>
            <dd>
              {isOwner
                ? 'Every declared action'
                : identity.scopes.length > 0
                  ? identity.scopes.map((scope) => <code key={scope}>{scope}</code>)
                  : '—'}
            </dd>
          </div>
          <div>
            <dt>Credential expires</dt>
            <dd>{formatDate(identity.expires_at)}</dd>
          </div>
        </dl>
      </section>

      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>{operators.length} principals</h2>
            <p>Credential values are never stored or served — only their digests.</p>
          </div>
          <button
            type="button"
            className="primary-button"
            disabled={!isOwner}
            title={isOwner ? undefined : 'Only the project owner may create a delegation'}
            onClick={() => setCreating(true)}
          >
            New delegation
          </button>
        </div>
        {operators.length === 0 ? (
          <EmptyState>No principals yet. Run `epor auth bootstrap` to mint the owner.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Principal</th>
                  <th>Role</th>
                  <th>Scopes</th>
                  <th>Expires</th>
                  <th className="action-cell">Actions</th>
                </tr>
              </thead>
              <tbody>
                {operators.map((operator) => (
                  <tr key={operator.id}>
                    <td className="job-name">
                      <strong>{operator.label}</strong>
                    </td>
                    <td>
                      <StatusPill status={operator.active ? operator.role : 'revoked'} />
                    </td>
                    <td>
                      {operator.role === 'owner'
                        ? 'every declared action'
                        : operator.scopes.map((scope) => <code key={scope}>{scope}</code>)}
                    </td>
                    <td className="numeric-cell">{formatDate(operator.expires_at)}</td>
                    <td className="action-cell">
                      <button
                        type="button"
                        className="ghost-button"
                        disabled={!isOwner || !operator.active}
                        onClick={() => void act(() => api.rotateOperator(operator.id))}
                      >
                        Rotate
                      </button>
                      <button
                        type="button"
                        className="danger-button"
                        disabled={!isOwner || !operator.active || operator.role === 'owner'}
                        title={
                          operator.role === 'owner'
                            ? 'The owner credential is rotated, never revoked'
                            : undefined
                        }
                        onClick={() => void act(() => api.revokeOperator(operator.id))}
                      >
                        Revoke
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {creating && (
        <DelegationForm
          actions={actions}
          onClose={() => setCreating(false)}
          onIssued={(next) => {
            setCreating(false)
            setIssued(next)
            void refresh()
          }}
        />
      )}
      {issued && <IssuedCredentialDialog issued={issued} onClose={() => setIssued(null)} />}
    </div>
  )
}

export function SignInDialog({
  onClose,
  onSignedIn,
}: {
  onClose: () => void
  onSignedIn: () => void
}) {
  const { dialogRef, onDialogKeyDown } = useDialogBehavior<HTMLFormElement>(onClose)
  const [token, setToken] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      await api.openSession(token)
      // Dropped from component state immediately: the browser keeps the
      // HttpOnly session cookie, never the bootstrap credential itself.
      setToken('')
      onSignedIn()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'That credential was not accepted')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <form
        ref={dialogRef}
        className="job-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="sign-in-title"
        aria-describedby="sign-in-help"
        tabIndex={-1}
        onSubmit={(event) => void submit(event)}
        onKeyDown={onDialogKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-head">
          <div>
            <h2 id="sign-in-title">Identify yourself</h2>
            <p id="sign-in-help">
              Exchanged for a 12-hour session held in an HttpOnly cookie. The credential itself is
              never stored by this page, and every session ends when the API restarts.
            </p>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <label className="form-field">
          <span>Credential</span>
          <input
            required
            type="password"
            aria-label="Credential"
            data-dialog-initial-focus
            autoComplete="off"
            value={token}
            onChange={(event) => setToken(event.target.value)}
          />
          <small>From `epor auth bootstrap`, or a delegation the owner issued you.</small>
        </label>
        {error && <div className="error-banner">{error}</div>}
        <div className="modal-actions">
          <button type="button" className="ghost-button" onClick={onClose}>
            Stay anonymous
          </button>
          <button className="primary-button" type="submit" disabled={busy || token === ''}>
            {busy ? 'Verifying…' : 'Sign in'}
          </button>
        </div>
      </form>
    </div>
  )
}
