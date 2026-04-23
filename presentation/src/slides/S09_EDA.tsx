import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { AccentCard } from '../components/AccentCard'
import { C } from '../design/tokens'

const findings = [
  { n: '01', color: C.red,    title: 'Severe Class Imbalance',
    body: 'Only ~3.5% of transactions are fraudulent. A naive majority-predictor achieves 96.5% accuracy — making recall-oriented metrics essential for meaningful evaluation.' },
  { n: '02', color: C.amber,  title: 'Temporal Fraud Signatures',
    body: 'Fraud rates vary significantly by hour of day and day of week. Time-series patterns reveal that fraudsters tend to operate at specific temporal windows.' },
  { n: '03', color: C.teal,   title: 'Widespread Missing Values',
    body: 'Missing values are unevenly distributed across feature groups (identity table has most). Handling missingness is central to model performance, not just a preprocessing detail.' },
  { n: '04', color: C.purple, title: 'Behavioral & Categorical Signals',
    body: 'Transaction amount, decimal patterns (3 decimal places = higher fraud risk), product category, and card/email domain features all show meaningful shifts between fraud and non-fraud.' },
]

export function S09_EDA() {
  return (
    <LightSlide title="Exploratory Data Analysis: Key Patterns" num={9}>
      <div className="flex flex-col h-full gap-2">
        <div className="grid grid-cols-2 gap-3 flex-1">
          {findings.map(({ n, color, title, body }, i) => (
            <motion.div key={n} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0, transition: { delay: i * 0.09 } }}>
              <AccentCard accent={color}>
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
          <strong>EDA TAKEAWAY:</strong> Fraud is not random noise — it leaves temporal, behavioral, and categorical signatures that machine learning can exploit.
        </motion.div>
      </div>
    </LightSlide>
  )
}
