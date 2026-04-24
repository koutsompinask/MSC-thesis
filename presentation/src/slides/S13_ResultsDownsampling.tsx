import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const before = [
  { model: 'XGBoost',  mc: C.amber,  recall: '0.671', auc: '0.896' },
  { model: 'LightGBM', mc: C.teal,   recall: '0.558', auc: '0.918' },
  { model: 'CatBoost', mc: C.purple, recall: '0.725', auc: '0.910' },
]
const after = [
  { model: 'XGBoost',  mc: C.amber,  recall: '0.703', delta: '+4.8%',  auc: '0.908' },
  { model: 'LightGBM', mc: C.teal,   recall: '0.626', delta: '+12.2%', auc: '0.917' },
  { model: 'CatBoost', mc: C.purple, recall: '0.724', delta: '-0.1%',  auc: '0.916' },
]

export function S13_ResultsDownsampling() {
  return (
    <LightSlide title="Results: Majority-Class Downsampling (1:5)" num={13}>
      <div className="flex flex-col h-full gap-4">
        <motion.div className="px-5 py-3 text-base font-bold text-white rounded" style={{ background: C.purple }}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          Downsampling improves XGBoost and CatBoost ROC-AUC, while LightGBM keeps ROC-AUC stable and gains PR-AUC
        </motion.div>

        <div className="flex gap-4 flex-1 items-stretch">
          {/* Before */}
          <motion.div className="flex-1 rounded overflow-hidden flex flex-col" style={{ border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.1 } }}>
            <div className="px-4 py-3 text-base font-bold text-white" style={{ background: C.textMid }}>BEFORE: Imbalanced</div>
            <div className="p-5 flex flex-col justify-center gap-4 flex-1">
              {before.map(({ model, mc, recall, auc }) => (
                <div key={model} className="rounded p-4 flex items-center gap-3" style={{ background: '#F1F7FF', border: '1px solid #e2e8f0' }}>
                  <div className="w-1.5 self-stretch rounded" style={{ background: mc }} />
                  <div>
                    <div className="font-bold text-lg" style={{ color: mc }}>{model}</div>
                    <div className="text-base" style={{ color: C.textMid }}>Recall: {recall}  |  ROC-AUC: {auc}</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Arrow */}
          <div className="flex flex-col items-center justify-center gap-2 flex-shrink-0 w-16">
            <div className="text-4xl font-black" style={{ color: C.teal }}>→</div>
            <div className="text-sm text-center font-bold" style={{ color: C.textMuted }}>1:5<br />DS</div>
          </div>

          {/* After */}
          <motion.div className="flex-1 rounded overflow-hidden flex flex-col" style={{ border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.2 } }}>
            <div className="px-4 py-3 text-base font-bold text-white" style={{ background: C.green }}>AFTER: 1:5 Downsampled</div>
            <div className="p-5 flex flex-col justify-center gap-4 flex-1">
              {after.map(({ model, mc, recall, delta, auc }) => (
                <div key={model} className="rounded p-4 flex items-center gap-3" style={{ background: '#F1F7FF', border: '1px solid #e2e8f0' }}>
                  <div className="w-1.5 self-stretch rounded" style={{ background: mc }} />
                  <div className="flex-1">
                    <div className="font-bold text-lg" style={{ color: mc }}>{model}</div>
                    <div className="flex items-center gap-3 text-base" style={{ color: C.textMid }}>
                      <span>Recall: {recall}</span>
                      <span className="px-2 py-0.5 rounded font-bold text-sm" style={{ background: C.greenPale, color: C.green }}>Change {delta}</span>
                      <span>AUC: {auc}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        <motion.div className="px-5 py-3 text-base rounded" style={{ background: C.navyDark, color: C.tealBright }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.4 } }}>
          Operational reading: controlled rebalancing changes recall and precision, while ROC-AUC remains strong across all three models.
        </motion.div>
      </div>
    </LightSlide>
  )
}
