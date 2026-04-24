import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { AccentCard } from '../components/AccentCard'
import { C } from '../design/tokens'

const findings = [
  { n: '01', color: C.red,    title: 'Severe Class Imbalance',
    body: 'Only 20,663 of 590,540 transactions are fraud (~3.5%). Accuracy alone becomes misleading, so ROC-AUC, PR-AUC, recall and threshold trade-offs matter.' },
  { n: '02', color: C.amber,  title: 'Temporal Fraud Signatures',
    body: 'Raw TransactionDT looked noisy, but deriving day_of_week and hour_of_day exposed clearer fraud-rate patterns. This motivated amount-by-hour/day context features.' },
  { n: '03', color: C.teal,   title: 'Widespread Missing Values',
    body: '214 features had >50% missing values. In fraud data, missing identity/device details can itself be informative, so only >99% missing columns were dropped.' },
  { n: '04', color: C.purple, title: 'Behavioral & Categorical Signals',
    body: 'Amount extremes, 3-decimal transactions (11.77% fraud), product/card/email shifts, and rare domains justified relational aggregates: _mean, _std, _rel, _ct and _freq.' },
]

export function S09_EDA() {
  return (
    <LightSlide title="Exploratory Data Analysis: Key Patterns" num={9}>
      <div className="flex flex-col h-full gap-2">
        <div className="grid grid-cols-2 gap-x-2 gap-y-3 flex-1 -mx-2">
          {findings.map(({ n, color, title, body }, i) => (
            <motion.div
              key={n}
              className="h-full"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0, transition: { delay: i * 0.09 } }}
            >
              <AccentCard accent={color} className="h-full">
                <div className="flex items-start gap-2.5">
                  <div
                    className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center font-bold text-xs text-white"
                    style={{ background: color }}
                  >
                    {n}
                  </div>
                  <div>
                    <div className="font-semibold text-base mb-1" style={{ color: C.textDark }}>{title}</div>
                    <div className="text-sm leading-relaxed" style={{ color: C.textMid }}>{body}</div>
                  </div>
                </div>
              </AccentCard>
            </motion.div>
          ))}
        </div>

        <motion.div
          className="px-4 py-2 text-sm rounded"
          style={{ background: C.navyDark, color: C.tealBright }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.5 } }}
        >
          <strong>EDA TAKEAWAY:</strong> The engineered features were not arbitrary — they encoded the exact temporal, amount, user, card and category deviations surfaced by EDA.
        </motion.div>
      </div>
    </LightSlide>
  )
}
