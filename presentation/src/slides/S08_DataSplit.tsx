import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { AccentCard } from '../components/AccentCard'
import { C } from '../design/tokens'

const callouts = [
  { color: C.teal,   t: 'No Data Leakage',     d: 'Future information never used to train on past data' },
  { color: C.green,  t: 'Simulates Deployment', d: 'Model validated on transactions it has never seen' },
  { color: C.purple, t: 'Robust Evaluation',    d: '5-fold CV provides stable performance estimates' },
]

export function S08_DataSplit() {
  return (
    <LightSlide title="Chronological Data Split & Time-Series Validation" num={8}>
      <div className="flex gap-5 h-full">
        {/* Left: timeline + CV */}
        <div className="flex-1 flex flex-col gap-3">
          <div className="text-xs font-bold" style={{ color: C.teal }}>
            → Chronological order (TransactionDT) — no shuffling at any stage
          </div>

          {/* Train / Val / Test bar */}
          <motion.div className="flex rounded overflow-hidden h-12" style={{ border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, scaleX: 0.8 }} animate={{ opacity: 1, scaleX: 1, transition: { duration: 0.5, ease: 'easeOut' } }}
          >
            <div className="flex-[72] flex items-center pl-3 font-bold text-xs text-white" style={{ background: C.navyMid }}>
              TRAINING SET — 72%
            </div>
            <div className="flex-[9] flex items-center justify-center font-bold text-xs text-white" style={{ background: C.teal }}>
              VAL<br />9%
            </div>
            <div className="flex-[19] flex items-center pl-2 font-bold text-xs text-white" style={{ background: C.green }}>
              TEST — 19%
            </div>
          </motion.div>
          <div className="flex justify-between text-xs" style={{ color: C.textMuted }}>
            <span>Jan 2017</span><span>~Oct 2017</span><span>Dec 2017</span>
          </div>

          {/* TimeSeriesCV folds */}
          <div>
            <div className="text-xs font-bold mb-2" style={{ color: C.textDark }}>
              Time-Series Cross-Validation (within training set)
            </div>
            <div className="flex flex-col gap-1.5">
              {[0, 1, 2, 3, 4].map((f) => (
                <motion.div key={f} className="flex h-5 rounded overflow-hidden"
                  initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.3 + f * 0.08 } }}
                >
                  {/* Prior folds */}
                  {f > 0 && (
                    <div style={{ flex: f, background: `${C.navyMid}88`, display: 'flex', alignItems: 'center' }} />
                  )}
                  {/* Active fold */}
                  <div
                    className="flex items-center px-1.5"
                    style={{ flex: 1, background: C.teal }}
                  >
                    <span className="text-white font-bold text-xs">Fold {f + 1}</span>
                  </div>
                  {/* Remaining empty */}
                  {f < 4 && <div style={{ flex: 4 - f, background: `${C.navyMid}22` }} />}
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: callout cards */}
        <div className="flex flex-col gap-2.5 w-52">
          {callouts.map(({ color, t, d }, i) => (
            <motion.div key={t} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.2 + i * 0.1 } }}>
              <AccentCard accent={color} title={t}>
                <div className="text-xs leading-snug" style={{ color: C.textMid }}>{d}</div>
              </AccentCard>
            </motion.div>
          ))}
        </div>
      </div>

      <motion.div
        className="mt-2 px-4 py-2 text-xs rounded"
        style={{ background: C.navyDark, color: C.tealBright }}
        initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.7 } }}
      >
        Chronological split is essential for fraud detection — random splitting would overestimate performance by leaking future fraud patterns into training.
      </motion.div>
    </LightSlide>
  )
}
