import { create } from 'zustand'

export type DemoStage =
  | 'idle'
  | 'selected'
  | 'stage1'
  | 'stage2'
  | 'stage3'
  | 'stage4'
  | 'stage5'
  | 'stage6'
  | 'done'

export type ExampleKey = 'clear_fraud' | 'clear_legit' | 'borderline'

export interface ShapEntry {
  feature: string
  value: number | null
  shap: number
  direction: 'positive' | 'negative'
}

export interface InferenceResult {
  probability: number
  shap_values: ShapEntry[]
  shap_base_value: number
  model_name: string
  roc_auc: number
  pr_auc: number
}

interface DemoStore {
  selectedExample: ExampleKey | null
  stage: DemoStage
  result: InferenceResult | null
  isLoading: boolean
  error: string | null
  selectExample: (key: ExampleKey) => void
  runInference: (features: Record<string, number | null>) => Promise<void>
  reset: () => void
}

const advance = (set: (s: Partial<DemoStore>) => void, stage: DemoStage, delay: number) =>
  new Promise<void>((res) => setTimeout(() => { set({ stage }); res() }, delay))

export const useDemoStore = create<DemoStore>()((set) => ({
  selectedExample: null,
  stage: 'idle',
  result: null,
  isLoading: false,
  error: null,

  selectExample: (key: ExampleKey) =>
    set({ selectedExample: key, stage: 'selected', result: null, error: null }),

  runInference: async (features: Record<string, number | null>) => {
    set({ isLoading: true, error: null, stage: 'stage1' })

    // Lazy import to avoid circular deps
    const fetchPromise = import('../demo/api').then((m) => m.predictExplain(features))

    await advance(set, 'stage2', 400)
    await advance(set, 'stage3', 400)
    await advance(set, 'stage4', 400)
    set({ stage: 'stage5' })

    let result: InferenceResult
    try {
      result = await fetchPromise
    } catch (e) {
      set({ isLoading: false, error: String(e), stage: 'selected' })
      return
    }

    await advance(set, 'stage6', 500)
    await new Promise<void>((res) => setTimeout(() => {
      set({ stage: 'done', result, isLoading: false })
      res()
    }, 400))
  },

  reset: () => set({ selectedExample: null, stage: 'idle', result: null, isLoading: false, error: null }),
}))
