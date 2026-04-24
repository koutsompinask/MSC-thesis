import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import { useDemoStore } from '../store/useDemoStore'
import type { ExampleKey, FeatureValue } from '../store/useDemoStore'
import { fetchExamples } from './api'
import type { ExampleData } from './api'
import { C } from '../design/tokens'

const CARD_CONFIGS: Record<ExampleKey, { label: string; desc: string; accent: string; icon: string }> = {
  clear_fraud: {
    label: 'High Risk',
    desc: 'Example selected near 100% model probability',
    accent: C.amber,
    icon: '⚠',
  },
  clear_legit: {
    label: 'Low Risk',
    desc: 'Example selected near 0% model probability',
    accent: C.teal,
    icon: '✓',
  },
  borderline: {
    label: 'Mid Risk',
    desc: 'Example selected near 50% model probability',
    accent: C.purple,
    icon: '~',
  },
}

export function TransactionCards() {
  const { selectedExample, stage, selectExample, runInference } = useDemoStore()
  const [examples, setExamples] = useState<ExampleData | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    fetchExamples()
      .then(setExamples)
      .catch((e) => setLoadError(String(e)))
  }, [])

  const isRunning = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'].includes(stage)
  const isDone = stage === 'done'

  return (
    <div className="flex flex-col gap-2 flex-shrink-0">
      <div className="text-xs font-bold uppercase tracking-widest" style={{ color: C.textMuted }}>
        Select a Transaction
      </div>

      {loadError && (
        <div className="text-sm p-3 rounded" style={{ background: '#FEE2E2', color: C.red }}>
          Could not load examples: {loadError}
        </div>
      )}

      <div className="flex gap-2">
        {(Object.keys(CARD_CONFIGS) as ExampleKey[]).map((key, i) => {
          const { label, accent, icon } = CARD_CONFIGS[key]
          const isSelected = selectedExample === key

          return (
            <motion.button
              key={key}
              disabled={isRunning}
              onClick={() => selectExample(key)}
              className="flex-1 rounded-md px-3 py-2 text-left transition-all cursor-pointer disabled:cursor-not-allowed"
              style={{
                background: isSelected ? `${accent}18` : C.navyMid,
                border: `1.5px solid ${isSelected ? accent : '#1A3A5C'}`,
                minHeight: '44px',
              }}
              whileHover={!isRunning ? { scale: 1.02 } : {}}
              animate={isSelected ? { opacity: 1, y: 0, scale: 1.03 } : { opacity: 1, y: 0, scale: 1 }}
              initial={{ opacity: 0, y: 8 }}
              transition={{ delay: i * 0.07 }}
            >
              <div className="flex items-center gap-2">
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                  style={{ background: accent, color: C.white }}
                >
                  {icon}
                </div>
                <span className="font-bold text-sm" style={{ color: isSelected ? accent : C.white }}>{label}</span>
              </div>
            </motion.button>
          )
        })}
      </div>

      {/* Run button */}
      {selectedExample && !isRunning && !isDone && examples && (
        <motion.button
          className="w-full py-2 rounded text-sm font-bold text-white"
          style={{ background: C.teal }}
          onClick={() => runInference(examples[selectedExample] as Record<string, FeatureValue>)}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          Run Inference →
        </motion.button>
      )}

      {isRunning && (
        <motion.div
          className="w-full py-2 rounded text-sm text-center font-medium"
          style={{ background: C.navyMid, color: C.tealBright, border: `1px solid ${C.teal}` }}
          animate={{ opacity: [1, 0.6, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          Running pipeline…
        </motion.div>
      )}
    </div>
  )
}
