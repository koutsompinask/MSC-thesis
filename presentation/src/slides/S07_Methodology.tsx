import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const steps = [
  { n: 1, title: 'EDA',                  desc: 'Analyze class imbalance, missingness, temporal structure, and fraud-related behavioral patterns', color: C.red },
  { n: 2, title: 'Preprocessing',        desc: 'Remove extreme-missingness fields and prepare the time-aware dataset for modeling',              color: C.teal },
  { n: 3, title: 'Feature Engineering',  desc: 'Create aggregate behavioral features such as mean, relative, avg, std, and frequency signals', color: C.purple },
  { n: 4, title: 'Initial Training',     desc: 'Train XGBoost, LightGBM, and CatBoost with Optuna tuning and time-series cross-validation',   color: C.navyMid },
  { n: 5, title: 'Downsampling',         desc: 'Test majority-class downsampling to improve learning under severe class imbalance',             color: C.green },
  { n: 6, title: 'Feature Reduction',    desc: 'Use cross-model SHAP importance agreement to retain the most informative compact feature set',   color: C.teal },
  { n: 7, title: 'Threshold Tuning',     desc: 'Adjust the classification threshold to study the fraud-capture versus false-positive trade-off', color: C.amber },
]

const rows = [steps.slice(0, 3), steps.slice(3)]

export function S07_Methodology() {
  return (
    <LightSlide title="Methodology Pipeline" num={7}>
      <div className="flex flex-col h-full justify-center gap-4">
        <div className="flex flex-col gap-3 flex-1">
          {rows.map((row, ri) => (
            <div
              key={ri}
              className="grid gap-3 flex-1"
              style={{
                gridTemplateColumns: `repeat(${row.length}, minmax(0, 1fr))`,
                width: ri === 0 ? '78%' : '100%',
                alignSelf: 'center',
              }}
            >
              {row.map(({ n, title, desc, color }) => (
                <div key={n} className="flex items-center">
                  <motion.div
                    className="flex-1 h-full rounded overflow-hidden"
                    style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 8px rgba(0,0,0,0.07)' }}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0, transition: { delay: (n - 1) * 0.1 } }}
                  >
                    <div className="py-3 flex items-center justify-center" style={{ background: color }}>
                      <div
                        className="w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm"
                        style={{ background: 'rgba(255,255,255,0.2)', color: C.white, border: `2px solid rgba(255,255,255,0.5)` }}
                      >
                        {n}
                      </div>
                    </div>
                    <div className="px-3 py-2.5">
                      <div className="font-bold text-sm text-center mb-1" style={{ color: C.textDark }}>{title}</div>
                      <div className="text-sm text-center leading-snug" style={{ color: C.textMid }}>{desc}</div>
                    </div>
                  </motion.div>
                </div>
              ))}
            </div>
          ))}
        </div>
        <motion.div
          className="text-sm text-center py-2 rounded" style={{ color: C.textMuted }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.6 } }}
        >
          Time-series cross-validation was used throughout to prevent data leakage and simulate real-world deployment conditions.
        </motion.div>
      </div>
    </LightSlide>
  )
}
