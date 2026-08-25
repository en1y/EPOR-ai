import { useState } from 'react'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { JobDetail, JobForm, JobsPage } from './App'
import { api } from './api'
import type { Job, JobTypeDefinition } from './types'

const job: Job = {
  id: '12345678-1234-1234-1234-123456789abc',
  type: 'system_probe',
  status: 'queued',
  spec: {},
  result: null,
  progress: 0,
  error_code: null,
  error_message: null,
  worker_id: null,
  retry_of_id: null,
  created_at: '2026-08-25T10:00:00Z',
  updated_at: '2026-08-25T10:00:00Z',
  started_at: null,
  finished_at: null,
  heartbeat_at: null,
  cancel_requested_at: null,
}

const researchDefinition: JobTypeDefinition = {
  type: 'research_sync',
  title: 'Research sync',
  description: 'Synchronize allowlisted research sources.',
  schema: {
    properties: {
      offline: { type: 'boolean', title: 'Offline', default: true },
      source_ids: {
        title: 'Source Ids',
        default: null,
        anyOf: [
          { type: 'array', items: { type: 'string' } },
          { type: 'null' },
        ],
      },
    },
  },
}

const evalDefinition: JobTypeDefinition = {
  type: 'tiny_eval',
  title: 'Tiny evaluation',
  description: 'Evaluate a tiny checkpoint.',
  schema: {
    required: ['checkpoint_path'],
    properties: {
      checkpoint_path: { type: 'string', title: 'Checkpoint Path' },
      model_config: {
        title: 'Model Config',
        default: null,
        anyOf: [{ type: 'string' }, { type: 'null' }],
      },
      max_batches: {
        title: 'Max Batches',
        default: null,
        anyOf: [
          { type: 'integer', minimum: 1, maximum: 10_000 },
          { type: 'null' },
        ],
      },
    },
  },
}

afterEach(() => vi.restoreAllMocks())

describe('schema-driven job form', () => {
  it('keeps nullable arrays empty and omits them from a research sync request', async () => {
    const user = userEvent.setup()
    const createJob = vi.spyOn(api, 'createJob').mockResolvedValue({
      ...job,
      type: 'research_sync',
    })

    render(<JobForm definitions={[researchDefinition]} onClose={() => undefined} onCreated={() => undefined} />)

    const sourceIds = screen.getByLabelText('Source Ids')
    expect(sourceIds).toHaveValue('')
    expect(sourceIds).toHaveAttribute('type', 'text')
    await user.click(screen.getByRole('button', { name: 'Queue job' }))

    await waitFor(() => {
      expect(createJob).toHaveBeenCalledWith('research_sync', { offline: true })
    })
  })

  it('renders nullable integers as empty number fields and omits untouched values', async () => {
    const user = userEvent.setup()
    const createJob = vi.spyOn(api, 'createJob').mockResolvedValue({
      ...job,
      type: 'tiny_eval',
    })

    render(<JobForm definitions={[evalDefinition]} onClose={() => undefined} onCreated={() => undefined} />)

    const maxBatches = screen.getByLabelText('Max Batches')
    expect(maxBatches).toHaveAttribute('type', 'number')
    expect(maxBatches).toHaveAttribute('min', '1')
    expect(maxBatches).toHaveAttribute('max', '10000')
    expect(maxBatches).toHaveValue(null)
    await user.type(screen.getByLabelText('Checkpoint Path'), 'runs/tiny/checkpoint.pt')
    await user.click(screen.getByRole('button', { name: 'Queue job' }))

    await waitFor(() => {
      expect(createJob).toHaveBeenCalledWith('tiny_eval', {
        checkpoint_path: 'runs/tiny/checkpoint.pt',
      })
    })
  })
})

describe('keyboard-accessible dialogs', () => {
  function FormHarness() {
    const [open, setOpen] = useState(false)
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>Open job form</button>
        {open && (
          <JobForm
            definitions={[researchDefinition]}
            onClose={() => setOpen(false)}
            onCreated={() => setOpen(false)}
          />
        )}
      </>
    )
  }

  it('sets initial focus, traps Tab, closes on Escape, and restores focus', async () => {
    const user = userEvent.setup()
    render(<FormHarness />)

    const trigger = screen.getByRole('button', { name: 'Open job form' })
    await user.click(trigger)
    const dialog = screen.getByRole('dialog', { name: 'Queue a local job' })
    expect(within(dialog).getByLabelText('Job type')).toHaveFocus()

    const close = within(dialog).getByRole('button', { name: 'Close' })
    const queue = within(dialog).getByRole('button', { name: 'Queue job' })
    queue.focus()
    await user.tab()
    expect(close).toHaveFocus()
    await user.tab({ shift: true })
    expect(queue).toHaveFocus()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  function DrawerHarness() {
    const [open, setOpen] = useState(false)
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>Open job detail</button>
        {open && <JobDetail job={job} onClose={() => setOpen(false)} onChanged={() => undefined} />}
      </>
    )
  }

  it('gives the job drawer a labelled modal boundary and restores its trigger', async () => {
    vi.spyOn(api, 'events').mockResolvedValue({ items: [], next_after: 0 })
    vi.spyOn(api, 'artifacts').mockResolvedValue([])
    const user = userEvent.setup()
    render(<DrawerHarness />)

    const trigger = screen.getByRole('button', { name: 'Open job detail' })
    await user.click(trigger)
    const dialog = screen.getByRole('dialog', { name: /system probe/i })
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(within(dialog).getByRole('button', { name: 'Close detail' })).toHaveFocus()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })
})

describe('job row keyboard activation', () => {
  it('opens once for Space and once for Enter', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()
    render(<JobsPage jobs={[job]} onSelect={onSelect} onNew={() => undefined} />)

    const row = screen.getByRole('button', { name: /Open system probe job/ })
    row.focus()
    await user.keyboard(' ')
    expect(onSelect).toHaveBeenCalledTimes(1)
    await user.keyboard('{Enter}')
    expect(onSelect).toHaveBeenCalledTimes(2)
  })
})
