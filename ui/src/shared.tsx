import { useCallback, useEffect, useRef } from 'react'

const focusableSelector = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

export function focusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(focusableSelector)).filter(
    (element) => element.getAttribute('aria-hidden') !== 'true',
  )
}

/**
 * Focus containment, Escape-to-close, and focus restoration for one dialog.
 * Shared so every dialog behaves identically for keyboard and screen-reader
 * users, rather than each one re-implementing a slightly different trap.
 */
export function useDialogBehavior<T extends HTMLElement>(onClose: () => void) {
  const dialogRef = useRef<T>(null)
  const restoreFocusRef = useRef<HTMLElement | null>(
    typeof document !== 'undefined' && document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null,
  )

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    const initialFocus =
      dialog.querySelector<HTMLElement>('[data-dialog-initial-focus]') ??
      focusableElements(dialog)[0] ??
      dialog
    initialFocus.focus()

    return () => {
      const restoreFocus = restoreFocusRef.current
      if (restoreFocus?.isConnected) restoreFocus.focus()
    }
  }, [])

  const onDialogKeyDown = useCallback(
    (event: React.KeyboardEvent<T>) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        onClose()
        return
      }
      if (event.key !== 'Tab') return

      const dialog = dialogRef.current
      if (!dialog) return
      const focusable = focusableElements(dialog)
      if (focusable.length === 0) {
        event.preventDefault()
        dialog.focus()
        return
      }

      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const active = document.activeElement
      if (!dialog.contains(active)) {
        event.preventDefault()
        ;(event.shiftKey ? last : first).focus()
      } else if (event.shiftKey && (active === first || active === dialog)) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && active === last) {
        event.preventDefault()
        first.focus()
      }
    },
    [onClose],
  )

  return { dialogRef, onDialogKeyDown }
}

export function titleCase(value: string): string {
  return value.replaceAll('_', ' ').replaceAll('-', ' ')
}

export function formatDate(value: string | null): string {
  if (!value) return '—'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

export function shortId(value: string): string {
  return value.slice(0, 8)
}

export function StatusPill({ status }: { status: string }) {
  const tone = ['succeeded', 'verified', 'ok', 'certified', 'ratified', 'consumed'].includes(status)
    ? 'positive'
    : ['failed', 'error', 'interrupted', 'refused'].includes(status)
      ? 'negative'
      : ['running', 'starting', 'cancelling', 'open', 'approved'].includes(status)
        ? 'active'
        : 'neutral'
  return <span className={`status-pill ${tone}`}>{titleCase(status)}</span>
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="empty-state">{children}</div>
}
