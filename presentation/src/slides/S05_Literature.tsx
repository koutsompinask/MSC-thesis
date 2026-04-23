import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const approaches = [
  { color: C.red,    tag: 'Traditional Supervised',
    models: 'Logistic Regression, SVM, Decision Trees',
    pro: 'Interpretable, fast, well-understood',
    con: 'Limited capacity to model complex interactions; poor recall on severe imbalance' },
  { color: C.amber,  tag: 'Anomaly Detection',
    models: 'Isolation Forest, One-Class SVM, LOF',
    pro: 'Works without labeled fraud — useful when labels are scarce',
    con: 'High false positive rates; not optimized for known fraud patterns in labeled data' },
  { color: C.purple, tag: 'Deep Learning',
    models: 'LSTM, Autoencoders, Transformers',
    pro: 'Powerful on sequential/unstructured data; learns latent representations',
    con: 'High-capacity, high-variance models can overfit tabular fraud data; also require large datasets, high compute, and are harder to explain' },
  { color: C.teal,   tag: 'Gradient Boosting Ensembles',
    models: 'XGBoost, LightGBM, CatBoost',
    pro: 'State-of-the-art on tabular data; they win many ML competitions, including the IEEE-CIS benchmark used in this thesis',
    con: 'Ensemble complexity requires SHAP or similar for interpretability' },
]

export function S05_Literature() {
  return (
    <LightSlide title="Literature Landscape: ML Approaches to Fraud Detection" num={5}>
      <div className="flex flex-col h-full gap-3">
        <div className="grid grid-cols-2 gap-4 flex-1">
          {approaches.map(({ color, tag, models, pro, con }, i) => (
            <motion.div
              key={tag}
              className="rounded overflow-hidden flex flex-col"
              style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0, transition: { delay: i * 0.09 } }}
            >
              <div className="px-3 py-2.5" style={{ background: color }}>
                <div className="font-bold text-white text-base">{tag}</div>
              </div>
              <div className="flex-1 px-4 py-3 flex flex-col gap-2">
                <div className="text-sm italic leading-relaxed" style={{ color: C.textMuted }}>{models}</div>
                <div className="flex items-start gap-2">
                  <div className="w-2.5 h-2.5 rounded-sm mt-1 flex-shrink-0" style={{ background: C.green }} />
                  <div className="text-sm leading-relaxed" style={{ color: C.textDark }}>{pro}</div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="w-2.5 h-2.5 rounded-sm mt-1 flex-shrink-0" style={{ background: C.red }} />
                  <div className="text-sm leading-relaxed" style={{ color: C.textMid }}>{con}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          className="rounded px-4 py-2.5 text-sm leading-relaxed"
          style={{ background: C.navyDark, color: C.tealBright }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.5 } }}
        >
          Literature consensus: gradient boosting remains the strongest choice for structured, labeled fraud data, offering competition-level performance on tabular benchmarks while preserving a practical path to explainability through SHAP.
        </motion.div>
      </div>
    </LightSlide>
  )
}
