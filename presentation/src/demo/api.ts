import type { InferenceResult } from '../store/useDemoStore'

const API_KEY = 'c1c58f5a-8f7c-4bdb-9d78-1c3b12c9f3f2'

export async function predictExplain(features: Record<string, number | null>): Promise<InferenceResult> {
  const res = await fetch('/api/predict_explain', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    body: JSON.stringify(features),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json()
}

export interface ExampleData {
  clear_fraud: Record<string, number | null>
  clear_legit: Record<string, number | null>
  borderline: Record<string, number | null>
}

export async function fetchExamples(): Promise<ExampleData> {
  const res = await fetch('/api/examples', {
    headers: { 'X-API-Key': API_KEY },
  })
  if (!res.ok) throw new Error(`Could not load examples: ${res.status}`)
  return res.json()
}
