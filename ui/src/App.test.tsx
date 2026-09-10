import { useState } from 'react'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App, { DataPage, JobDetail, JobForm, JobsPage, TrainingPage } from './App'
import { api } from './api'
import type {
  Capabilities,
  DataSummary,
  Health,
  Job,
  JobEvent,
  JobTypeDefinition,
  ResearchCatalog,
} from './types'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((next) => { resolve = next })
  return { promise, resolve }
}

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
  submitted_by_id: null,
  submitted_by_role: null,
  admission_covenant_sha256: null,
  escalation_id: null,
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

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

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

describe('job detail polling', () => {
  const event = (sequence: number, kind: string): JobEvent => ({
    sequence,
    kind,
    payload: { message: `event ${sequence}` },
    created_at: `2026-08-25T10:00:0${sequence}Z`,
  })

  it('deduplicates and sorts out-of-order responses without regressing the cursor', async () => {
    vi.useFakeTimers()
    const older = deferred<{ items: JobEvent[]; next_after: number }>()
    const newer = deferred<{ items: JobEvent[]; next_after: number }>()
    const events = vi.spyOn(api, 'events')
      .mockImplementationOnce(() => older.promise)
      .mockImplementationOnce(() => newer.promise)
      .mockResolvedValue({ items: [], next_after: 3 })
    vi.spyOn(api, 'artifacts').mockResolvedValue([])

    render(<JobDetail job={job} onClose={() => undefined} onChanged={() => undefined} />)
    expect(events).toHaveBeenNthCalledWith(1, job.id, 0)

    await act(async () => { await vi.advanceTimersByTimeAsync(1_800) })
    expect(events).toHaveBeenNthCalledWith(2, job.id, 0)

    await act(async () => {
      newer.resolve({ items: [event(3, 'job.third'), event(2, 'job.second')], next_after: 3 })
      await newer.promise
    })
    await act(async () => {
      older.resolve({ items: [event(2, 'job.second-repeat'), event(1, 'job.first')], next_after: 2 })
      await older.promise
    })

    expect(
      screen.getAllByRole('listitem').map((item) => item.querySelector('strong')?.textContent),
    ).toEqual(['first', 'second', 'third'])

    await act(async () => { await vi.advanceTimersByTimeAsync(1_800) })
    expect(events).toHaveBeenNthCalledWith(3, job.id, 3)
  })
})

describe('progress accessibility', () => {
  it('exposes table progress as a named percentage progressbar', () => {
    render(<JobsPage jobs={[{ ...job, progress: 0.426 }]} onSelect={() => undefined} onNew={() => undefined} />)

    const progress = screen.getByRole('progressbar', { name: /system probe job 12345678 progress/i })
    expect(progress).toHaveAttribute('aria-valuemin', '0')
    expect(progress).toHaveAttribute('aria-valuemax', '100')
    expect(progress).toHaveAttribute('aria-valuenow', '43')
    expect(progress).toHaveAttribute('aria-valuetext', '43%')
  })

  it('exposes drawer progress and clamps invalid values to its range', () => {
    vi.spyOn(api, 'events').mockResolvedValue({ items: [], next_after: 0 })
    vi.spyOn(api, 'artifacts').mockResolvedValue([])
    render(<JobDetail job={{ ...job, progress: 1.2 }} onClose={() => undefined} onChanged={() => undefined} />)

    const progress = screen.getByRole('progressbar', { name: /system probe job progress/i })
    expect(progress).toHaveAttribute('aria-valuenow', '100')
    expect(progress).toHaveAttribute('aria-valuetext', '100%')
    expect(progress.firstElementChild).toHaveStyle({ width: '100%' })
  })
})

describe('API health polling', () => {
  const health: Health = { status: 'ok', database: 'ok', version: '0.0.1' }
  const capabilities: Capabilities = {
    bind_host: '127.0.0.1',
    allowed_origin: 'http://127.0.0.1:5173',
    job_types: ['system_probe', 'research_sync', 'tiny_train', 'tiny_eval'],
    cancellation: true,
    event_transport: 'polling',
    arbitrary_commands: false,
    arbitrary_url_fetching: false,
    filesystem_browser: false,
    hardware: {},
  }
  const research: ResearchCatalog = {
    schema_version: 1,
    available: true,
    catalog_path: 'research/catalog.yaml',
    sources: [],
    count: 0,
    error: null,
  }
  const dataSummary: DataSummary = {
    schema_version: 1,
    registrations: 0,
    documents: 0,
    admitted: 0,
    quarantined: 0,
    tombstones: 0,
    builds: 0,
    integrity_errors: [],
    latest_builds: {},
    sources: [],
  }

  it('marks the API unavailable after a failed refresh and recovers later', async () => {
    vi.useFakeTimers()
    const healthRequest = vi.spyOn(api, 'health')
      .mockResolvedValueOnce(health)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValue(health)
    vi.spyOn(api, 'capabilities').mockResolvedValue(capabilities)
    vi.spyOn(api, 'research').mockResolvedValue(research)
    vi.spyOn(api, 'data').mockResolvedValue(dataSummary)
    vi.spyOn(api, 'models').mockResolvedValue([])
    vi.spyOn(api, 'jobTypes').mockResolvedValue([])
    vi.spyOn(api, 'jobs').mockResolvedValue([])
    // The shell asks who you are on mount; anonymous is the default answer.
    vi.spyOn(api, 'identity').mockRejectedValue(new Error('anonymous'))
    vi.spyOn(api, 'actions').mockRejectedValue(new Error('unavailable'))
    vi.spyOn(api, 'covenant').mockRejectedValue(new Error('unavailable'))

    render(<App />)
    await act(async () => { await Promise.resolve() })
    expect(screen.getByText('EPOR 0.0.1')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })
    expect(screen.getByText('EPOR offline')).toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })
    expect(screen.getByText('EPOR 0.0.1')).toBeInTheDocument()
    expect(healthRequest).toHaveBeenCalledTimes(3)
  })
})

describe('Data provenance page', () => {
  it('surfaces rights state and queues typed data jobs', async () => {
    const user = userEvent.setup()
    const onNew = vi.fn()
    const summary: DataSummary = {
      schema_version: 1,
      registrations: 1,
      documents: 3,
      admitted: 2,
      quarantined: 1,
      tombstones: 1,
      builds: 1,
      integrity_errors: [],
      latest_builds: { 'fixture-v1': 'a'.repeat(64) },
      sources: [{
        id: 'fixture',
        title: 'Fixture corpus',
        owner_or_steward: 'Fixture steward',
        license_id: 'fixture-only',
        allowed_uses: ['research', 'train'],
        sensitive_content_risk: 'low',
      }],
    }
    render(<DataPage summary={summary} onNew={onNew} />)

    expect(screen.getByText('Fixture corpus')).toBeInTheDocument()
    expect(screen.getByText('fixture-only')).toBeInTheDocument()
    expect(screen.getByText('Quarantined')).toBeInTheDocument()
    expect(screen.getAllByText('1', { selector: '.metric-card strong' }).length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: 'Ingest' }))
    expect(onNew).toHaveBeenCalledWith('data_ingest')
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

const trainDefinition: JobTypeDefinition = {
  type: 'tiny_train',
  title: 'Tiny pretraining',
  description: 'Run the deterministic CPU reference trainer.',
  schema: {
    properties: {
      max_steps: { type: 'integer', title: 'Max Steps', default: 20, minimum: 1, maximum: 1000 },
    },
  },
}

const trainRun: Job = {
  ...job,
  id: 'aaaaaaaa-1234-1234-1234-123456789abc',
  type: 'tiny_train',
  status: 'succeeded',
  progress: 1,
  result: {
    run_id: 'run-1',
    steps_completed: 20,
    final_loss: 2.5,
    checkpoint_path: 'aaaaaaaa/tiny-train/checkpoints/step-20.pt',
  },
}

describe('training page', () => {
  it('reports run metrics and hands a checkpoint to evaluation', async () => {
    const user = userEvent.setup()
    const onEvaluate = vi.fn()
    const onSelect = vi.fn()
    render(
      <TrainingPage
        jobs={[trainRun, { ...job, id: 'bbbbbbbb', type: 'system_probe' }]}
        onSelect={onSelect}
        onTrain={() => undefined}
        onEvaluate={onEvaluate}
      />,
    )

    const row = screen.getByRole('button', { name: /Open tiny train run/ }).closest('tr')!
    expect(within(row).getByText('20')).toBeInTheDocument()
    expect(within(row).getByText('2.5000')).toBeInTheDocument()
    // Non-training work stays on the Jobs page.
    expect(screen.queryByRole('button', { name: /system probe/i })).not.toBeInTheDocument()

    await user.click(within(row).getByRole('button', { name: 'Evaluate' }))
    expect(onEvaluate).toHaveBeenCalledWith('aaaaaaaa/tiny-train/checkpoints/step-20.pt')
    // Evaluating must not also open the run drawer behind the form.
    expect(onSelect).not.toHaveBeenCalled()
  })
})

describe('prefilled job form', () => {
  it('seeds the requested type with supplied values and keeps schema defaults', async () => {
    const user = userEvent.setup()
    const createJob = vi.spyOn(api, 'createJob').mockResolvedValue({ ...job, type: 'tiny_eval' })

    render(
      <JobForm
        definitions={[trainDefinition, evalDefinition]}
        initialType="tiny_eval"
        initialValues={{ checkpoint_path: 'run-1/checkpoints/step-20.pt' }}
        onClose={() => undefined}
        onCreated={() => undefined}
      />,
    )

    expect(screen.getByLabelText('Checkpoint Path')).toHaveValue('run-1/checkpoints/step-20.pt')
    await user.click(screen.getByRole('button', { name: 'Queue job' }))

    await waitFor(() => {
      expect(createJob).toHaveBeenCalledWith('tiny_eval', {
        checkpoint_path: 'run-1/checkpoints/step-20.pt',
      })
    })
  })

  it('drops the seed when the operator switches to another job type', async () => {
    const user = userEvent.setup()
    render(
      <JobForm
        definitions={[evalDefinition, trainDefinition]}
        initialType="tiny_eval"
        initialValues={{ checkpoint_path: 'run-1/checkpoints/step-20.pt' }}
        onClose={() => undefined}
        onCreated={() => undefined}
      />,
    )

    await user.selectOptions(screen.getByLabelText('Job type'), 'tiny_train')
    expect(screen.queryByLabelText('Checkpoint Path')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Max Steps')).toHaveValue(20)
  })
})
