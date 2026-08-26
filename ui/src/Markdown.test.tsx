import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Markdown, resolveSlug, safeHref } from './Markdown'

const mermaidRender = vi.fn().mockResolvedValue({ svg: '<svg data-testid="diagram"></svg>' })
vi.mock('mermaid', () => ({
  default: { initialize: vi.fn(), render: (...args: unknown[]) => mermaidRender(...args) },
}))

describe('markdown rendering', () => {
  it('escapes raw HTML instead of injecting it', () => {
    const { container } = render(
      <Markdown markdown={'Before\n\n<img src=x onerror="alert(1)">\n\nAfter'} />,
    )

    expect(container.querySelector('img')).toBeNull()
    expect(container.textContent).toContain('<img src=x onerror="alert(1)">')
  })

  it('drops scripted link protocols but keeps real ones', () => {
    const { container } = render(
      <Markdown markdown={'[bad](javascript:alert(1)) and [good](https://example.test/paper)'} />,
    )

    const links = [...container.querySelectorAll('a')]
    expect(links).toHaveLength(1)
    expect(links[0]).toHaveAttribute('href', 'https://example.test/paper')
    expect(links[0]).toHaveAttribute('rel', 'noreferrer noopener')
    expect(container.textContent).toContain('bad')
  })

  it('renders a mermaid fence through the lazily imported diagram engine', async () => {
    const { container } = render(<Markdown markdown={'```mermaid\ngraph TD\nA-->B\n```'} />)

    await waitFor(() => expect(container.querySelector('figure.diagram')).not.toBeNull())
    expect(mermaidRender).toHaveBeenCalledWith(expect.any(String), 'graph TD\nA-->B')
  })

  it('does not let document text forge a diagram marker', () => {
    const { container } = render(
      <Markdown markdown={'Inline <!--epor-diagram--> and\n\n<!--epor-diagram-->\n'} />,
    )

    expect(container.querySelector('figure.diagram')).toBeNull()
    expect(container.querySelector('pre.mermaid-source')).toBeNull()
    expect(container.textContent).toContain('<!--epor-diagram-->')
  })

  it('keeps the drawn diagram across re-renders of the reader', async () => {
    const markdown = '```mermaid\ngraph TD\nA-->B\n```'
    const { container, rerender } = render(<Markdown markdown={markdown} />)

    await waitFor(() => expect(container.querySelector('figure.diagram')).not.toBeNull())
    // The parent polls on a timer, so a redraw must not discard the diagram.
    rerender(<Markdown markdown={markdown} slug="docs/ARCHITECTURE.md" />)
    expect(container.querySelector('figure.diagram')).not.toBeNull()
    expect(container.querySelector('pre.mermaid-source')).toBeNull()
  })

  it('falls back to the diagram source when the engine refuses it', async () => {
    mermaidRender.mockRejectedValueOnce(new Error('unsupported diagram type'))
    const { container } = render(<Markdown markdown={'```mermaid\nnot a diagram\n```'} />)

    await waitFor(() => expect(mermaidRender).toHaveBeenCalled())
    expect(container.querySelector('figure.diagram')).toBeNull()
    expect(container.querySelector('pre.mermaid-source')?.textContent).toBe('not a diagram')
  })

  it('keeps relative document links inside the reader', async () => {
    const user = userEvent.setup()
    const onNavigate = vi.fn()
    render(
      <Markdown
        markdown={'[Roadmap](ROADMAP.md) and [Catalog](../research/catalog.yaml)'}
        slug="docs/README.md"
        onNavigate={onNavigate}
      />,
    )

    await user.click(screen.getByText('Roadmap'))
    expect(onNavigate).toHaveBeenCalledWith('docs/ROADMAP.md')

    // A non-Markdown target is left to the browser rather than swallowed.
    await user.click(screen.getByText('Catalog'))
    expect(onNavigate).toHaveBeenCalledTimes(1)
  })
})

describe('link helpers', () => {
  it('refuses every scheme except http, https, and mailto', () => {
    expect(safeHref('https://example.test')).toBe('https://example.test')
    expect(safeHref('mailto:person@example.test')).toBe('mailto:person@example.test')
    expect(safeHref('docs/ROADMAP.md')).toBe('docs/ROADMAP.md')
    expect(safeHref('#section')).toBe('#section')
    expect(safeHref('javascript:alert(1)')).toBeNull()
    expect(safeHref('JavaScript:alert(1)')).toBeNull()
    expect(safeHref('data:text/html,<script>')).toBeNull()
  })

  it('resolves relative markdown targets against the current slug', () => {
    expect(resolveSlug('docs/README.md', 'ROADMAP.md')).toBe('docs/ROADMAP.md')
    expect(resolveSlug('docs/README.md', 'research/anthropic.md')).toBe('docs/research/anthropic.md')
    expect(resolveSlug('docs/research/README.md', '../ROADMAP.md')).toBe('docs/ROADMAP.md')
    expect(resolveSlug('docs/README.md', 'ROADMAP.md#gates')).toBe('docs/ROADMAP.md#gates')
    expect(resolveSlug('docs/README.md', '../research/catalog.yaml')).toBeNull()
    expect(resolveSlug('docs/README.md', 'https://example.test/x.md')).toBeNull()
  })
})
