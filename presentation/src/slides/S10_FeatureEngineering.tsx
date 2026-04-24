import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const cats = [
  { color: C.teal,   title: 'User Anchor',
    items: ['Simulated uid from card1 + addr1 + account-age proxy', 'Turns anonymized rows into repeatable behavior groups', 'Enables each transaction to be compared with its own baseline'] },
  { color: C.purple, title: 'Numeric Deviations',
    items: ['Group numeric fields by uid, card type, product and time context', 'Create _mean, _std and _rel features', 'Capture transactions that are unusual for that entity/context'] },
  { color: C.green,  title: 'Categorical Frequency',
    items: ['Group categorical fields by uid', 'Create value-count and frequency features (_ct, _freq)', 'Expose rare device, email, match-flag and identity combinations'] },
  { color: C.amber,  title: 'Selection Logic',
    items: ['Generate broad relational feature space first', 'Let model importance remove redundant signals', 'Several engineered aggregates ranked among the strongest predictors'] },
]

export function S10_FeatureEngineering() {
  return (
    <LightSlide title="Feature Engineering Strategy" num={10}>
      <div className="grid grid-cols-2 gap-4 h-full">
        {cats.map(({ color, title, items }, i) => (
          <motion.div
            key={title}
            className="rounded overflow-hidden"
            style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 8px rgba(0,0,0,0.07)' }}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0, transition: { delay: i * 0.09 } }}
          >
            <div className="px-3 py-2.5 font-bold text-base text-white" style={{ background: color }}>
              {title}
            </div>
            <div className="px-3 py-2.5 flex flex-col gap-2">
              {items.map((item) => (
                <div key={item} className="flex items-start gap-2">
                  <div className="w-2 h-2 rounded-sm mt-1 flex-shrink-0" style={{ background: color }} />
                  <div className="text-sm leading-snug" style={{ color: C.textMid }}>{item}</div>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </LightSlide>
  )
}
