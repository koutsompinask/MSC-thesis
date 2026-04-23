import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { AccentCard } from '../components/AccentCard'
import { C } from '../design/tokens'

const callouts = [
  { color: C.teal,   t: 'No Data Leakage',  d: 'Future information never used to train on past data' },
  { color: C.green,  t: '80/20 Holdout',    d: 'First 80% used for training, final 20% reserved as unseen test data' },
  { color: C.purple, t: '3-Fold CV',        d: 'Validation performed inside the training set with expanding time-series folds' },
]

const folds = [
  { label: 'CV Split 1', train: 2, val: 1, future: 2 },
  { label: 'CV Split 2', train: 3, val: 1, future: 1 },
  { label: 'CV Split 3', train: 4, val: 1, future: 0 },
]

export function S08_DataSplit() {
  return (
    <LightSlide title="Chronological Data Split & Time-Series Validation" num={8}>
      <div className="flex gap-5 h-full">
        <div className="flex-1 flex flex-col gap-3">
          <div className="text-sm font-bold" style={{ color: C.teal }}>
            → Chronological order (`TransactionDT`) — no shuffling and no standalone validation split
          </div>

          <motion.div
            className="flex rounded overflow-hidden h-12"
            style={{ border: '1px solid #e2e8f0' }}
            initial={{ opacity: 0, scaleX: 0.8 }}
            animate={{ opacity: 1, scaleX: 1, transition: { duration: 0.5, ease: 'easeOut' } }}
          >
            <div
              className="flex items-center pl-3 font-bold text-sm text-white"
              style={{ background: C.navyMid, flex: 4 }}
            >
              TRAINING SET — 80%
            </div>
            <div
              className="flex items-center justify-center font-bold text-sm text-white"
              style={{ background: C.green, flex: 1 }}
            >
              TEST — 20%
            </div>
          </motion.div>
          <div className="flex justify-between text-sm" style={{ color: C.textMuted }}>
            <span>Past</span><span>Future</span>
          </div>

          <div>
            <div className="text-sm font-bold mb-2" style={{ color: C.textDark }}>
              Time-Series Cross-Validation (within training set)
            </div>
            <div className="flex flex-col gap-3 py-3">
              {folds.map(({ label, train, val, future }, f) => (
                <motion.div
                  key={label}
                  className="flex items-center gap-3"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0, transition: { delay: 0.25 + f * 0.08 } }}
                >
                  <div className="w-20 text-sm font-semibold" style={{ color: C.textDark }}>
                    {label}
                  </div>
                  <div className="flex-1 flex h-10 rounded overflow-hidden" style={{ border: '1px solid #dbe4f0' }}>
                    <div style={{ flex: train, background: '#8FD3FF' }} />
                    <div style={{ flex: val, background: '#F97316' }} />
                    {future > 0 && <div style={{ flex: future, background: `${C.navyMid}20` }} />}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="w-full pr-3">
            <motion.div
              className="mt-1 px-4 py-2 text-sm rounded"
              style={{ background: C.navyDark, color: C.tealBright }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, transition: { delay: 0.55 } }}
            >
              Chronological evaluation is critical in fraud detection — random splitting would leak future patterns into training.
            </motion.div>
          </div>
        </div>

        <div className="flex flex-col gap-2.5 w-52">
          {callouts.map(({ color, t, d }, i) => (
            <motion.div key={t} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.2 + i * 0.1 } }}>
              <AccentCard accent={color} title={t}>
                <div className="text-sm leading-snug" style={{ color: C.textMid }}>{d}</div>
              </AccentCard>
            </motion.div>
          ))}
        </div>
      </div>
    </LightSlide>
  )
}
