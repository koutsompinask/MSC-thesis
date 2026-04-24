import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const models = [
  { name: 'XGBoost',  mc: C.amber,
    before: { rec: '0.678', prec: '0.300' }, after: { rec: '0.922', prec: '0.085' } },
  { name: 'LightGBM', mc: C.teal,
    before: { rec: '0.661', prec: '0.348' }, after: { rec: '0.895', prec: '0.110' } },
  { name: 'CatBoost', mc: C.purple,
    before: { rec: '0.720', prec: '0.298' }, after: { rec: '0.938', prec: '0.076' } },
]

export function S16_ResultsThreshold() {
  return (
    <LightSlide title="Results: Decision Threshold Tuning" num={16}>
      <div className="flex flex-col h-full gap-2">
        <motion.div className="px-4 py-2 text-sm font-bold text-white rounded" style={{ background: C.amber }}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          Lowering the reduced-feature model threshold to 0.1 sharply increases fraud recall at the cost of precision
        </motion.div>

        <div className="flex gap-3 flex-1">
          {models.map(({ name, mc, before, after }, i) => (
            <motion.div key={name} className="flex-1 rounded overflow-hidden flex flex-col"
              style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
              initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0, transition: { delay: i * 0.1 } }}
            >
              <div className="py-2.5 text-center font-bold text-base text-white" style={{ background: mc }}>{name}</div>

              <div className="flex-1 flex flex-col items-center justify-center gap-3 p-3">
                {/* Before */}
                <div className="w-full rounded p-2.5" style={{ background: '#F1F7FF', border: '1px solid #e2e8f0' }}>
                  <div className="text-sm font-bold mb-1" style={{ color: C.textMuted }}>Threshold 0.5</div>
                  <div className="text-sm" style={{ color: C.textMid }}>Recall: {before.rec}</div>
                  <div className="text-sm" style={{ color: C.textMid }}>Precision: {before.prec}</div>
                </div>

                <div className="text-center">
                  <div className="text-xl font-black" style={{ color: C.amber }}>↓</div>
                  <div className="text-sm" style={{ color: C.textMuted }}>threshold → 0.1</div>
                </div>

                {/* After */}
                <div className="w-full rounded p-2.5" style={{ background: C.amberPale, border: `1px solid ${C.amber}` }}>
                  <div className="text-sm font-bold mb-1" style={{ color: C.amber }}>Threshold 0.1</div>
                  <div className="text-sm font-bold" style={{ color: C.green }}>Recall: {after.rec}</div>
                  <div className="text-sm font-bold" style={{ color: C.red }}>Precision: {after.prec}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div className="px-4 py-2 text-sm rounded" style={{ background: C.navyDark, color: C.tealBright }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.4 } }}>
          The thesis frames the threshold as an operational lever tied to fraud-loss tolerance and available review capacity.
        </motion.div>
      </div>
    </LightSlide>
  )
}
