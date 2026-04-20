import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const features = [
  { name: 'card1  (card identifier)',          val: 0.92, color: C.teal },
  { name: 'TransactionAmt_mean  (behavioral)', val: 0.87, color: C.teal },
  { name: 'TransactionAmt',                   val: 0.80, color: '#06B6D4' },
  { name: 'C1  (transaction count)',          val: 0.74, color: C.navyMid },
  { name: 'D1  (time delta)',                  val: 0.68, color: C.navyMid },
  { name: 'addr1  (billing address)',          val: 0.62, color: C.navyMid },
  { name: 'card1_addr1  (interaction)',        val: 0.55, color: C.purple },
]

export function S11_SHAP() {
  return (
    <LightSlide title="SHAP: Feature Importance & Model Explainability" num={11}>
      <div className="flex gap-5 h-full">
        {/* Left: feature bars */}
        <div className="flex-1 rounded overflow-hidden" style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}>
          <div className="px-3 py-2.5 font-bold text-sm text-white" style={{ background: C.navyDark }}>
            Top Predictors by SHAP Importance (Cross-Model)
          </div>
          <div className="px-3 py-2.5 flex flex-col gap-2">
            {features.map(({ name, val, color }, i) => (
              <div key={name} className="flex items-center gap-2">
                <div className="w-4 text-xs font-bold text-right flex-shrink-0" style={{ color: C.textMuted }}>{i + 1}</div>
                <div className="flex-1 relative h-7 rounded overflow-hidden" style={{ background: '#DDE8F4' }}>
                  <motion.div
                    className="absolute left-0 top-0 h-full flex items-center pl-2"
                    style={{ background: color, width: `${val * 100}%` }}
                    initial={{ width: 0 }}
                    animate={{ width: `${val * 100}%`, transition: { delay: 0.2 + i * 0.07, duration: 0.5, ease: 'easeOut' } }}
                  >
                    <span className="text-white font-bold text-xs truncate">{name}</span>
                  </motion.div>
                </div>
                <div className="text-xs font-bold w-8 text-right flex-shrink-0" style={{ color: C.textDark }}>
                  {Math.round(val * 100)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: explainability cards */}
        <div className="w-52 flex flex-col gap-3">
          {[
            { color: C.teal, title: 'Global Explainability',
              body: 'SHAP ranks the features driving fraud detection across all transactions — showing which variables the model relies on most. Enables audit, compliance reporting, and feature selection.' },
            { color: C.purple, title: 'Local Explainability',
              body: 'SHAP assigns a contribution score to each feature for every individual prediction — answering "why was this specific transaction flagged?" This enables analyst review and appeals processes.' },
          ].map(({ color, title, body }, i) => (
            <motion.div
              key={title}
              className="rounded overflow-hidden flex-1"
              style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
              initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.2 + i * 0.1 } }}
            >
              <div className="px-3 py-2 font-bold text-sm text-white" style={{ background: color }}>{title}</div>
              <div className="px-3 py-2.5 text-xs leading-relaxed" style={{ color: C.textMid }}>{body}</div>
            </motion.div>
          ))}
        </div>
      </div>

      <motion.div
        className="mt-2 px-4 py-2 text-xs font-bold text-white rounded"
        style={{ background: C.teal }}
        initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.6 } }}
      >
        SHAP explainability is already delivered in this thesis — both global rankings and per-prediction attribution scores were computed for all three models.
      </motion.div>
    </LightSlide>
  )
}
