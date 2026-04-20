import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const configs = [
  { n: 'C1', color: C.teal,   title: 'Baseline — Full Features, Imbalanced Data',
    desc: 'Train on original data without any rebalancing. Establishes the performance ceiling for raw gradient boosting.' },
  { n: 'C2', color: C.purple, title: 'Majority-Class Downsampling (1:5 Ratio)',
    desc: 'Reduce legitimate transactions to 1:5 ratio. Tests whether controlled rebalancing helps models learn fraud patterns better.' },
  { n: 'C3', color: C.green,  title: 'Reduced Feature Set (SHAP-Based Selection)',
    desc: 'Select features based on cross-model SHAP importance agreement (top 30% per model, ≥2 models → 215 features). Tests competitive compact models.' },
  { n: 'C4', color: C.amber,  title: 'Decision Threshold Tuning (0.1)',
    desc: 'Lower threshold from default 0.5 to 0.1. Demonstrates the operational precision–recall trade-off.' },
]

const models = [
  { name: 'XGBoost',  color: C.amber  },
  { name: 'LightGBM', color: C.teal   },
  { name: 'CatBoost', color: C.purple },
]
const metrics = ['ROC-AUC (primary)', 'PR-AUC', 'Precision / Recall', 'F1-Score']

export function S12_ExperimentalSetup() {
  return (
    <LightSlide title="Experimental Configurations" num={12}>
      <div className="flex gap-5 h-full">
        {/* Left: configs */}
        <div className="flex-1 flex flex-col gap-2">
          {configs.map(({ n, color, title, desc }, i) => (
            <motion.div
              key={n}
              className="flex overflow-hidden rounded"
              style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 6px rgba(0,0,0,0.06)' }}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0, transition: { delay: i * 0.09 } }}
            >
              <div className="w-10 flex-shrink-0 flex items-center justify-center font-bold text-sm text-white" style={{ background: color }}>{n}</div>
              <div className="flex-1 py-2 px-3">
                <div className="font-semibold text-xs mb-0.5" style={{ color: C.textDark }}>{title}</div>
                <div className="text-xs" style={{ color: C.textMid }}>{desc}</div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Right: models + metrics */}
        <div className="w-44 flex flex-col gap-3">
          <motion.div className="rounded overflow-hidden" style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.15 } }}
          >
            <div className="px-3 py-2 font-bold text-xs text-white" style={{ background: C.navyDark }}>MODELS COMPARED</div>
            <div className="px-3 py-2 flex flex-col gap-2">
              {models.map(({ name, color }) => (
                <div key={name} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: color }} />
                  <span className="font-bold text-sm" style={{ color: C.textDark }}>{name}</span>
                </div>
              ))}
            </div>
          </motion.div>
          <motion.div className="rounded overflow-hidden" style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.25 } }}
          >
            <div className="px-3 py-2 font-bold text-xs text-white" style={{ background: C.navyDark }}>EVALUATION METRICS</div>
            <div className="px-3 py-2 flex flex-col gap-1.5">
              {metrics.map((m) => (
                <div key={m} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-sm flex-shrink-0" style={{ background: C.teal }} />
                  <span className="text-xs" style={{ color: C.textMid }}>{m}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </LightSlide>
  )
}
