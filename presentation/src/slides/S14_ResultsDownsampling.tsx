import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const before = [
  { model: 'XGBoost',  mc: C.amber,  recall: '0.671', auc: '0.896' },
  { model: 'LightGBM', mc: C.teal,   recall: '0.456', auc: '0.918' },
  { model: 'CatBoost', mc: C.purple, recall: '0.461', auc: '0.910' },
]
const after = [
  { model: 'XGBoost',  mc: C.amber,  recall: '0.703', delta: '+4.8%',  auc: '0.908' },
  { model: 'LightGBM', mc: C.teal,   recall: '0.626', delta: '+37.3%', auc: '0.917' },
  { model: 'CatBoost', mc: C.purple, recall: '0.757', delta: '+64.2%', auc: '0.916' },
]

export function S14_ResultsDownsampling() {
  return (
    <LightSlide title="Results: Majority-Class Downsampling (1:5)" num={14}>
      <div className="flex flex-col h-full gap-2">
        <motion.div className="px-4 py-2 text-xs font-bold text-white rounded" style={{ background: C.purple }}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          Downsampling improves fraud sensitivity across all models without significantly collapsing ROC-AUC performance
        </motion.div>

        <div className="flex gap-3 flex-1 items-stretch">
          {/* Before */}
          <motion.div className="flex-1 rounded overflow-hidden" style={{ border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.1 } }}>
            <div className="px-3 py-2 text-xs font-bold text-white" style={{ background: C.textMid }}>BEFORE: Imbalanced</div>
            <div className="p-2 flex flex-col gap-1.5">
              {before.map(({ model, mc, recall, auc }) => (
                <div key={model} className="rounded p-2 flex items-center gap-2" style={{ background: '#F1F7FF', border: '1px solid #e2e8f0' }}>
                  <div className="w-1 self-stretch rounded" style={{ background: mc }} />
                  <div>
                    <div className="font-bold text-xs" style={{ color: mc }}>{model}</div>
                    <div className="text-xs" style={{ color: C.textMid }}>Recall: {recall}  |  ROC-AUC: {auc}</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Arrow */}
          <div className="flex flex-col items-center justify-center gap-1 flex-shrink-0 w-14">
            <div className="text-2xl font-black" style={{ color: C.teal }}>→</div>
            <div className="text-xs text-center font-medium" style={{ color: C.textMuted }}>1:5<br />DS</div>
          </div>

          {/* After */}
          <motion.div className="flex-1 rounded overflow-hidden" style={{ border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.2 } }}>
            <div className="px-3 py-2 text-xs font-bold text-white" style={{ background: C.green }}>AFTER: 1:5 Downsampled</div>
            <div className="p-2 flex flex-col gap-1.5">
              {after.map(({ model, mc, recall, delta, auc }) => (
                <div key={model} className="rounded p-2 flex items-center gap-2" style={{ background: '#F1F7FF', border: '1px solid #e2e8f0' }}>
                  <div className="w-1 self-stretch rounded" style={{ background: mc }} />
                  <div className="flex-1">
                    <div className="font-bold text-xs" style={{ color: mc }}>{model}</div>
                    <div className="flex items-center gap-2 text-xs" style={{ color: C.textMid }}>
                      <span>Recall: {recall}</span>
                      <span className="px-1.5 py-0.5 rounded font-bold text-xs" style={{ background: C.greenPale, color: C.green }}>↑ {delta}</span>
                      <span>AUC: {auc}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        <motion.div className="px-4 py-2 text-xs rounded" style={{ background: C.navyDark, color: C.tealBright }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.4 } }}>
          Operational reading: controlled rebalancing substantially increases fraud capture with only marginal ROC-AUC cost.
        </motion.div>
      </div>
    </LightSlide>
  )
}
