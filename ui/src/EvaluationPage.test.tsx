import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { EvaluationPage } from './EvaluationPage'
import { api } from './api'
import type {
  EvaluationReportList,
  EvaluationReportSummary,
  EvaluationReportView,
  ReportComparison,
} from './types'

const summary: EvaluationReportSummary = {
  report_id: 'a'.repeat(64),
  suite_id: 'epor-safety-behaviour',
  suite_version: 1,
  scope: 'harness-fixture',
  claim_boundary: 'Harness-fixture result: not evidence of any model capability.',
  model_kind: 'kernel-checkpoint',
  model_id: 'epor-tiny',
  checkpoint_step: 60,
  parameter_count: 12345,
  task_count: 4,
  skipped_task_count: 1,
  primary_metrics: [
    { task_id: 'safety-preference', task_type: 'safety-preference', metric: 'preference_norm', value: 0.5, scope: 'harness-fixture' },
  ],
  code_executed: false,
}

const other: EvaluationReportSummary = {
  ...summary,
  report_id: 'b'.repeat(64),
  suite_id: 'epor-harness-fixtures',
  model_id: 'epor-tiny-b',
}

const listing: EvaluationReportList = {
  schema_version: 1,
  reports: [summary, other],
  integrity_errors: ['c'.repeat(64) + ': evaluation report ID does not match its content'],
}

const metric = (name: string, value: number | null, higherIsBetter = true) => ({
  name,
  value,
  scope: 'harness-fixture',
  higher_is_better: higherIsBetter,
})

const view: EvaluationReportView = {
  report_id: summary.report_id,
  suite_id: summary.suite_id,
  suite_version: 1,
  scope: 'harness-fixture',
  claim_boundary: summary.claim_boundary,
  model: { kind: 'kernel-checkpoint', model_id: 'epor-tiny', checkpoint_step: 60, parameter_count: 12345, tokenizer_id: 'd'.repeat(64) },
  data: [],
  skipped_tasks: [{ task_id: 'fixture-code', reason: 'code execution not enabled' }],
  summary,
  tasks: [
    {
      task_id: 'safety-preference',
      task_type: 'safety-preference',
      format: 'safety-pref-v1',
      primary_metric: 'preference_norm',
      metrics: [
        metric('preference_norm', 0.5),
        metric('preference_norm.category.direct-harm', 0.75),
        metric('preference_norm.category.shutdown-and-correction', null),
      ],
      item_count: 12,
      settings: { claim_note: 'A passing score on a small proxy model proves nothing about safety.' },
      calibration: null,
      sandbox: null,
    },
    {
      task_id: 'fixture-multiple-choice',
      task_type: 'multiple-choice',
      format: 'mc-v1',
      primary_metric: 'accuracy_norm',
      metrics: [metric('accuracy_norm', 0.25), metric('ece', 0.125, false)],
      item_count: 4,
      settings: {},
      calibration: {
        method: 'max-probability-equal-width',
        probability_source: 'softmax of length-normalized log-likelihoods',
        bins: 2,
        items: 4,
        expected_calibration_error: 0.125,
        brier_score: 0.7,
        reliability: [
          { lower: 0, upper: 0.5, count: 3, mean_confidence: 0.4, accuracy: 0.333 },
          { lower: 0.5, upper: 1, count: 1, mean_confidence: 0.6, accuracy: 0 },
        ],
      },
      sandbox: null,
    },
    {
      task_id: 'contamination',
      task_type: 'contamination',
      format: 'ngram-contamination-v1',
      primary_metric: 'flagged_rate',
      metrics: [metric('flagged_rate', 0.1, false), metric('flagged_items', 2, false), metric('assessed_items', 20)],
      item_count: 20,
      settings: {},
      calibration: null,
      sandbox: null,
    },
    {
      task_id: 'memorization',
      task_type: 'memorization',
      format: 'memorization-v1',
      primary_metric: 'extraction_gap',
      metrics: [
        metric('extraction_gap', 0.25, false),
        metric('train.exact_suffix_rate', 0.5, false),
        metric('heldout.exact_suffix_rate', 0.25, false),
      ],
      item_count: 8,
      settings: {},
      calibration: null,
      sandbox: null,
    },
  ],
}

const comparison: ReportComparison = {
  baseline_report_id: summary.report_id,
  candidate_report_id: other.report_id,
  warnings: ['reports come from different suites or suite versions'],
  metric_deltas: [
    { task_id: 'safety-preference', name: 'preference_norm', baseline: 0.5, candidate: 0.75, delta: 0.25, higher_is_better: true },
  ],
  bootstraps: [
    {
      task_id: 'safety-preference',
      paired_items: 12,
      unmatched_items: 0,
      baseline_mean: 0.5,
      candidate_mean: 0.75,
      mean_delta: 0.25,
      ci_lower: 0.1,
      ci_upper: 0.4,
      confidence: 0.95,
      resamples: 1000,
      seed: 0,
    },
  ],
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('EvaluationPage', () => {
  it('lists verified reports, names failures, and queues only typed jobs', async () => {
    vi.spyOn(api, 'evaluationReports').mockResolvedValue(listing)
    const onNew = vi.fn()
    render(<EvaluationPage onNew={onNew} checkpointPath="job-1/kernel" tokenizedId={'d'.repeat(64)} />)

    const table = await screen.findByRole('table', { name: 'Stored evaluation reports' })
    expect(within(table).getByText('epor-safety-behaviour v1')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Verification failed')
    expect(screen.getByText(/evaluation jobs always skip them/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Run suite' }))
    expect(onNew).toHaveBeenCalledWith('eval_suite', { checkpoint_path: 'job-1/kernel' })
    await userEvent.click(screen.getByRole('button', { name: 'Scan contamination' }))
    expect(onNew).toHaveBeenCalledWith('eval_contamination', { training_data: 'd'.repeat(64) })
  })

  it('shows scoped metrics, calibration, contamination, memorization, and the safety claim note', async () => {
    vi.spyOn(api, 'evaluationReports').mockResolvedValue({ ...listing, integrity_errors: [] })
    vi.spyOn(api, 'evaluationReport').mockResolvedValue(view)
    render(<EvaluationPage onNew={vi.fn()} checkpointPath={null} tokenizedId={null} />)

    const table = await screen.findByRole('table', { name: 'Stored evaluation reports' })
    await userEvent.click(within(table).getAllByRole('button', { name: 'View' })[0])

    const detail = await screen.findByRole('region', { name: `Evaluation report ${'a'.repeat(8)}` })
    expect(within(detail).getByText('Claim boundary.')).toBeInTheDocument()
    expect(within(detail).getByText(/code execution not enabled/)).toBeInTheDocument()

    const note = within(detail).getByRole('note')
    expect(note).toHaveTextContent('Not evidence of safety')
    expect(note).toHaveTextContent('proves nothing about safety')
    const categories = within(detail).getByRole('table', { name: 'safety-preference G10 categories' })
    expect(within(categories).getByText('direct harm')).toBeInTheDocument()
    expect(within(categories).getByText('0.7500')).toBeInTheDocument()
    expect(within(categories).getByText('—')).toBeInTheDocument()

    const bins = within(detail).getByRole('table', { name: 'fixture-multiple-choice reliability bins' })
    expect(within(bins).getAllByRole('row')).toHaveLength(3)
    expect(within(detail).getByText(/ECE 0.1250/)).toBeInTheDocument()

    expect(within(detail).getByText('2 of 20 assessed item(s) flagged')).toBeInTheDocument()
    expect(within(detail).getByText(/train 0.500 vs held-out 0.250/)).toBeInTheDocument()
  })

  it('compares two reports and surfaces comparability warnings', async () => {
    vi.spyOn(api, 'evaluationReports').mockResolvedValue({ ...listing, integrity_errors: [] })
    const compare = vi.spyOn(api, 'compareReports').mockResolvedValue(comparison)
    render(<EvaluationPage onNew={vi.fn()} checkpointPath={null} tokenizedId={null} />)

    await screen.findByRole('table', { name: 'Stored evaluation reports' })
    const button = screen.getByRole('button', { name: 'Compare' })
    expect(button).toBeDisabled()
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Baseline report' }), summary.report_id)
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Candidate report' }), other.report_id)
    await userEvent.click(button)

    await waitFor(() => expect(compare).toHaveBeenCalledWith(summary.report_id, other.report_id))
    const panel = await screen.findByRole('region', { name: 'Report comparison' })
    expect(within(panel).getByRole('alert')).toHaveTextContent('different suites')
    expect(within(panel).getByText('0.1000 to 0.4000 at 95%')).toBeInTheDocument()
  })
})
