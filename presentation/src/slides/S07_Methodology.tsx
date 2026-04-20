import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const steps = [
  { n: 1, title: 'Preprocessing',       desc: 'Handle missingness, encode categorical variables, prepare temporal features', color: C.teal },
  { n: 2, title: 'Feature Engineering', desc: 'Build behavioral aggregates: mean, relative, avg, std, frequency features',  color: C.purple },
  { n: 3, title: 'Model Training',       desc: 'Train XGBoost, LightGBM, CatBoost with time-aware cross-validation',        color: C.navyMid },
  { n: 4, title: 'Evaluation',           desc: 'Compare ROC-AUC, PR-AUC, Precision, Recall, F1 across all configurations',  color: C.green },
  { n: 5, title: 'Sensitivity Tests',    desc: 'Downsampling, reduced features, and decision threshold tuning experiments',  color: C.amber },
]

export function S07_Methodology() {
  return (
    <LightSlide title="Methodology Pipeline" num={7}>
      <div className="flex flex-col h-full justify-center gap-4">
        <div className="flex items-start gap-2">
          {steps.map(({ n, title, desc, color }, i) => (
            <div key={n} className="flex items-center flex-1">
              <motion.div
                className="flex-1 rounded overflow-hidden"
                style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 8px rgba(0,0,0,0.07)' }}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0, transition: { delay: i * 0.1 } }}
              >
                <div className="py-4 flex items-center justify-center" style={{ background: color }}>
                  <div
                    className="w-9 h-9 rounded-full flex items-center justify-center font-bold text-sm"
                    style={{ background: 'rgba(255,255,255,0.2)', color: C.white, border: `2px solid rgba(255,255,255,0.5)` }}
                  >
                    {n}
                  </div>
                </div>
                <div className="px-3 py-2.5">
                  <div className="font-bold text-xs text-center mb-1" style={{ color: C.textDark }}>{title}</div>
                  <div className="text-xs text-center leading-snug" style={{ color: C.textMid }}>{desc}</div>
                </div>
              </motion.div>
              {i < steps.length - 1 && (
                <div className="text-lg font-bold mx-1 flex-shrink-0" style={{ color: '#94A3B8' }}>›</div>
              )}
            </div>
          ))}
        </div>
        <motion.div
          className="text-xs text-center py-2 rounded" style={{ color: C.textMuted }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.6 } }}
        >
          Time-series cross-validation was used throughout to prevent data leakage and simulate real-world deployment conditions.
        </motion.div>
      </div>
    </LightSlide>
  )
}
