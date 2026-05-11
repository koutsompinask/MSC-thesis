import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const columns = [
  { color: C.navyMid, title: 'What Stayed Consistent',
    items: [
      'LightGBM was the most stable ROC-AUC performer across the main model configurations',
      'Engineered behavioral features repeatedly appeared as the strongest predictors',
      'ROC-AUC remained stable even when the experimental setup changed significantly',
      'Gradient boosting models proved robust to different data conditions',
    ]},
  { color: C.green, title: 'What Improved',
    items: [
      'Downsampling improved XGBoost and CatBoost ROC-AUC while LightGBM gained PR-AUC',
      'Reduced features preserved strong discrimination with lower complexity',
      'Threshold reduction sharply increased fraud capture, with recall near or above 0.9',
      'Feature reduction preserved near-full discriminatory power with a compact feature set',
    ]},
  { color: C.amber, title: 'What This Means',
    items: [
      'Fraud detection is an operational optimization problem, not just a modeling task',
      'Model choice, sampling strategy, and threshold jointly affect deployment behavior',
      'ROC-AUC, PR-AUC, precision, recall and F1 each reveal different trade-offs',
      'A compact reduced-feature model can retain strong performance with lower complexity',
    ]},
]

export function S18_Synthesis() {
  return (
    <LightSlide title="Cross-Experiment Synthesis" num={18}>
      <div className="flex flex-col h-full gap-2">
        <div className="flex gap-3 flex-1">
          {columns.map(({ color, title, items }, ci) => (
            <motion.div key={title} className="flex-1 rounded overflow-hidden"
              style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 8px rgba(0,0,0,0.07)' }}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0, transition: { delay: ci * 0.1 } }}
            >
              <div className="px-3 py-2.5 font-bold text-base text-white" style={{ background: color }}>{title}</div>
              <div className="px-3 py-2.5 flex flex-col gap-2">
                {items.map((item, ii) => (
                  <div key={ii} className="flex items-start gap-2">
                    <div className="w-2 h-2 rounded-sm mt-1 flex-shrink-0" style={{ background: color }} />
                    <div className="text-sm leading-snug" style={{ color: C.textMid }}>{item}</div>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div className="px-4 py-2 text-sm font-bold rounded" style={{ background: C.navyDark, color: C.tealBright }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.4 } }}>
          Strong fraud detection depends not on the model alone, but on the full decision pipeline: features, balance, and threshold.
        </motion.div>
      </div>
    </LightSlide>
  )
}
