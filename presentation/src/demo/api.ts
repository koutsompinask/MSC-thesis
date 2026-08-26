import type { FeatureValue, InferenceResult } from '../store/useDemoStore'

// Requests go through the dev-server proxy at /api, which attaches the
// X-API-Key header on the way to the backend. Nothing secret lives here.

export async function predictExplain(features: Record<string, FeatureValue>): Promise<InferenceResult> {
  const res = await fetch('/api/predict_explain', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
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
  clear_fraud: Record<string, FeatureValue>
  clear_legit: Record<string, FeatureValue>
  borderline: Record<string, FeatureValue>
}

export async function fetchExamples(): Promise<ExampleData> {
  const res = await fetch('/api/examples')
  if (!res.ok) throw new Error(`Could not load examples: ${res.status}`)
  return res.json()
}
