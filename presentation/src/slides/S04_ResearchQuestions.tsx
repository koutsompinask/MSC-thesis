import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const rqs = [
  { n: 'RQ 1', color: C.teal,   q: 'Model Discrimination',
    t: 'How effectively do gradient boosting models (XGBoost, LightGBM, CatBoost) discriminate fraudulent from legitimate transactions under ROC-AUC evaluation?' },
  { n: 'RQ 2', color: C.purple, q: 'Impact of Downsampling',
    t: 'How does majority-class downsampling (1:5 ratio) affect model performance compared to training on the original imbalanced data distribution?' },
  { n: 'RQ 3', color: C.green,  q: 'Feature Set Reduction',
    t: 'Can a reduced feature set, selected through cross-model feature importance agreement, preserve the discriminatory power of the full feature space?' },
  { n: 'RQ 4', color: C.amber,  q: 'Threshold Tuning Effects',
    t: 'How does altering the decision threshold change fraud detection behavior — specifically the recall–precision trade-off relevant to operational deployment?' },
]

export function S04_ResearchQuestions() {
  return (
    <LightSlide title="Research Questions" num={4}>
      <div className="grid grid-cols-2 gap-4 h-full">
        {rqs.map(({ n, color, q, t }, i) => (
          <motion.div
            key={n}
            className="rounded overflow-hidden flex flex-col"
            style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0, transition: { delay: i * 0.1 } }}
          >
            <div className="flex items-center px-3 py-2.5" style={{ background: color }}>
              <span className="font-bold text-white text-xs mr-3">{n}</span>
              <span className="font-bold text-white text-sm">{q}</span>
            </div>
            <div className="flex-1 px-4 py-3 text-xs leading-relaxed" style={{ color: C.textMid }}>
              {t}
            </div>
          </motion.div>
        ))}
      </div>
    </LightSlide>
  )
}
