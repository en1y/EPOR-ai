import { useEffect, useMemo, useState } from 'react'
import { Markdown } from './Markdown'
import { api } from './api'
import type { DocumentRead, DocumentSummary } from './types'

const GROUP_TITLES: Record<string, string> = {
  '': 'Project root',
  docs: 'Documentation',
  'docs/research': 'Research dossiers',
}

function groupTitle(group: string): string {
  return GROUP_TITLES[group] ?? group
}

export function DocsPage({ initialSlug = 'docs/README.md' }: { initialSlug?: string }) {
  const [index, setIndex] = useState<DocumentSummary[]>([])
  const [slug, setSlug] = useState(initialSlug)
  const [document, setDocument] = useState<DocumentRead | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let disposed = false
    api
      .documents()
      .then((items) => {
        if (disposed) return
        setIndex(items)
        if (!items.some((item) => item.slug === slug) && items.length > 0) {
          setSlug(items[0].slug)
        }
      })
      .catch((caught: unknown) => {
        if (!disposed) setError(caught instanceof Error ? caught.message : 'Could not list documents')
      })
    return () => {
      disposed = true
    }
    // The index is a stable server-side listing; it is fetched once per visit.
  }, [])

  useEffect(() => {
    let disposed = false
    const [path, fragment] = slug.split('#')
    setError('')
    api
      .document(path)
      .then((next) => {
        if (disposed) return
        setDocument(next)
        const anchor = fragment ? window.document.getElementById(fragment) : null
        anchor?.scrollIntoView()
      })
      .catch((caught: unknown) => {
        if (disposed) return
        setDocument(null)
        setError(caught instanceof Error ? caught.message : 'Could not read that document')
      })
    return () => {
      disposed = true
    }
  }, [slug])

  const groups = useMemo(() => {
    const byGroup = new Map<string, DocumentSummary[]>()
    for (const item of index) {
      byGroup.set(item.group, [...(byGroup.get(item.group) ?? []), item])
    }
    return [...byGroup.entries()]
  }, [index])

  const activeSlug = slug.split('#')[0]

  return (
    <div className="docs-layout">
      <nav className="docs-index" aria-label="Documentation index">
        {groups.map(([group, items]) => (
          <section key={group}>
            <h2>{groupTitle(group)}</h2>
            <ul>
              {items.map((item) => (
                <li key={item.slug}>
                  <button
                    type="button"
                    className={item.slug === activeSlug ? 'docs-link current' : 'docs-link'}
                    aria-current={item.slug === activeSlug ? 'page' : undefined}
                    onClick={() => setSlug(item.slug)}
                  >
                    {item.title}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
        {index.length === 0 && !error && <p className="muted-line">Loading the index…</p>}
      </nav>
      <article className="docs-reader panel">
        {error && <div className="error-banner">{error}</div>}
        {document && (
          <>
            <header className="docs-reader-head">
              <h1>{document.title}</h1>
              <code>{document.slug}</code>
            </header>
            <Markdown markdown={document.markdown} slug={document.slug} onNavigate={setSlug} />
          </>
        )}
      </article>
    </div>
  )
}
