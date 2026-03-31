export type PredictRequest = {
  sequence: string
  top_k: number
  max_seq_len?: number | null
}

export type TermScore = {
  term: string
  probability: number
  name?: string
  scaled_confidence: number
  uac: number
  lower_bound?: number
  upper_bound?: number
  variance?: number
}

export type PredictionAspect = {
  aspect_label: string
  top_predictions: TermScore[]
}

export interface PredictResponse {
  protein_name: string
  sequence_type: string
  uncertainty: string
  entropy: number
  complexity_score: number
  explanation: {
    reason: string
    interpretation: string
    confidence_adjustment: string
    quantitative_decomposition?: {
      base_probability: number
      calibration_adjustment: number
      entropy_penalty: number
      diversity_bonus: number
      final_estimated_confidence: number
    }
    counterfactual?: string
    explanation_confidence?: number
    prediction_reliability_score?: number
    expected_accuracy_range?: string
  }
  confidence_decomposition?: {
    final_confidence: number
    total_reduction_pct: number
    complexity_score: number
    steps: Array<{
      stage: string
      value: number
      delta: number
      reason: string
    }>
  }
  prediction_withheld?: boolean
  withhold_reason?: string
  token_attributions?: number[]
  results: Record<string, PredictionAspect>
  model_name: string
  validation_accuracy_percent: number
  primary_summary: string
  notes: string[]
}

export interface AccuracyStats {
  micro_f1: number
  macro_f1: number
  top_k_accuracy: {
    top_1: number
    top_3: number
    top_5: number
  }
  coverage: number
  rejection_rate?: number
  optimal_threshold?: number
  accuracy_before_rejection: number
  accuracy_after_rejection: number
  accuracy_by_entropy: { bin: string; accuracy: number; count: number }[]
  accuracy_by_confidence: { bin: string; accuracy: number; count: number }[]
  accuracy_by_uac: { bin: string; accuracy: number; count: number }[]
  risk_coverage: { threshold: number; coverage: number; accuracy: number }[]
  micro_f1_ci?: [number, number]
  data_source?: string
  evaluation_metadata?: {
    total_samples: number
    accepted_samples: number
    rejected_samples: number
    go_terms_evaluated: number
    model: string
    parameters: string
  }
  integrity?: {
    validated: boolean
    threshold_source: string
    message: string
  }
}

export interface MetricsSummary {
  fmax: number
  auprc: number
  ece: number
  brier: number
  micro_f1?: number
  macro_f1?: number
  model?: string
  dataset_size?: number
  go_terms?: number
  data_source: 'results_file' | 'fallback'
}

export interface HealthStatus {
  status: string
  model_loaded: boolean
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type BenchmarkData = Record<string, any>

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

// ── Resilient fetch with retry, timeout, and structured error handling ───────

// Production deployments (e.g. Render) may have cold starts of 30s+
const IS_PRODUCTION = !!import.meta.env.VITE_API_URL
const DEFAULT_TIMEOUT_MS = IS_PRODUCTION ? 60000 : 8000
const DEFAULT_RETRIES = IS_PRODUCTION ? 4 : 3
const RETRY_DELAY_MS = 1500

async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  retries = DEFAULT_RETRIES,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<Response> {
  let lastError: Error | null = null

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`[API] Fetching ${url} (attempt ${attempt}/${retries})`)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

      const res = await fetch(url, {
        ...options,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)
      return res
    } catch (err: unknown) {
      const error = err instanceof Error ? err : new Error(String(err))
      lastError = error

      const isTimeout = error.name === 'AbortError'
      const isNetwork = error.message === 'Failed to fetch' || error.message.includes('NetworkError')

      console.warn(
        `[API] Attempt ${attempt}/${retries} failed: ${isTimeout ? 'Timeout' : error.message}`,
      )

      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * attempt))
      }

      // Don't retry non-network errors (e.g. CORS won't suddenly fix itself)
      if (!isTimeout && !isNetwork && attempt === 1) {
        break
      }
    }
  }

  // All retries exhausted — throw meaningful error
  if (lastError?.name === 'AbortError') {
    throw new Error(
      IS_PRODUCTION
        ? 'Backend not responding (timeout). The server may be waking up — please retry in 30 seconds.'
        : 'Backend not responding (timeout). Is the server running on port 8000?'
    )
  }
  if (lastError?.message === 'Failed to fetch' || lastError?.message.includes('NetworkError')) {
    throw new Error(
      IS_PRODUCTION
        ? 'Cannot connect to backend. The service may be starting up — please wait and retry.'
        : 'Cannot connect to backend. Please start the server: uvicorn backend.app:app --port 8000'
    )
  }
  throw lastError ?? new Error('Unknown API error')
}

// ── API Functions ────────────────────────────────────────────────────────────

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetchWithRetry(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  }, 2, 30000) // Prediction can take longer, 2 retries, 30s timeout

  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const data = await res.json()
      if (data?.detail?.error) {
        message = String(data.detail.error)
      } else if (typeof data?.detail === 'string') {
        message = data.detail
      } else if (Array.isArray(data?.detail)) {
        message = data.detail.map((d: Record<string, string>) => d.msg).join('; ')
      }
    } catch {
      // ignore json parse failure
    }
    throw new Error(message)
  }
  const data = await res.json()
  if (!validatePredictResponse(data)) {
    throw new Error('Invalid prediction response from backend')
  }
  return data
}

function validatePredictResponse(data: unknown): data is PredictResponse {
  if (!data || typeof data !== 'object') return false
  const d = data as Record<string, unknown>
  return (
    typeof d.protein_name === 'string' &&
    typeof d.uncertainty === 'string' &&
    typeof d.results === 'object' &&
    d.results !== null
  )
}

export async function checkHealth(): Promise<HealthStatus> {
  console.log('[API] Checking backend health...')
  const res = await fetchWithRetry(`${API_URL}/health`)
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`)
  }
  return (await res.json()) as HealthStatus
}

export async function fetchMetrics(): Promise<MetricsSummary> {
  console.log('[API] Fetching metrics...')
  const res = await fetchWithRetry(`${API_URL}/metrics`)
  if (!res.ok) {
    throw new Error(`Failed to fetch metrics (${res.status})`)
  }
  const data = await res.json()
  if (!data || typeof data.fmax !== 'number') {
    throw new Error('Invalid metrics response: missing required fields')
  }
  return data as MetricsSummary
}

export async function fetchAccuracyStats(): Promise<AccuracyStats> {
  console.log('[API] Fetching accuracy stats...')
  const res = await fetchWithRetry(`${API_URL}/accuracy_stats`)

  if (!res.ok) {
    let detail = `Failed to fetch accuracy stats (${res.status})`
    try {
      const data = await res.json()
      if (typeof data?.detail === 'string') {
        detail = data.detail
      }
    } catch {
      // ignore
    }
    throw new Error(detail)
  }

  const data = await res.json()

  // Validate essential fields exist
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid response: empty or malformed data')
  }
  if (!data.top_k_accuracy || !data.risk_coverage) {
    throw new Error('Invalid response: missing required fields')
  }

  return data as AccuracyStats
}

export async function fetchBenchmarks(): Promise<BenchmarkData> {
  const res = await fetchWithRetry(`${API_URL}/benchmarks`)
  if (!res.ok) throw new Error('Failed to load benchmarks')
  return res.json()
}

export interface DatasetInfo {
  total_proteins: number
  train: number
  validation: number
  test: number
  go_terms: number
  model: string
  parameters: string
}

export async function fetchDatasetInfo(): Promise<DatasetInfo> {
  console.log('[API] Fetching dataset info...')
  const res = await fetchWithRetry(`${API_URL}/dataset_info`)
  if (!res.ok) throw new Error('Failed to fetch dataset info')
  return res.json() as Promise<DatasetInfo>
}

export interface ModelInfo {
  trained_on_samples: number
  last_trained: string
  dataset_version: string
  calibration_msg: string
  is_synced: boolean
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  console.log('[API] Fetching model info...')
  const res = await fetchWithRetry(`${API_URL}/model_info`)
  if (!res.ok) throw new Error('Failed to fetch model info')
  return res.json() as Promise<ModelInfo>
}
