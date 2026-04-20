import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const cats = [
  { color: C.teal,   title: 'Behavioral Aggregates',
    items: ['User-level transaction means (_mean)', 'Relative deviation from user baseline (_rel)', 'Activity frequency signals (_freq, _std, _avg)'] },
  { color: C.purple, title: 'Temporal Patterns',
    items: ['Hour of day & day of week context', 'Delta-time between transactions', 'Sequence-aware user behavior signals'] },
  { color: C.green,  title: 'Entity Interactions',
    items: ['Amount × card type features', 'Amount × product category signals', 'Cross-feature transaction relationships'] },
  { color: C.amber,  title: 'Why It Matters',
    items: ['Captures hidden transaction habits', 'Improves discrimination beyond raw fields', 'Engineered features ranked top predictors across all 3 models'] },
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
            <div className="px-3 py-2.5 font-bold text-sm text-white" style={{ background: color }}>
              {title}
            </div>
            <div className="px-3 py-2.5 flex flex-col gap-2">
              {items.map((item) => (
                <div key={item} className="flex items-start gap-2">
                  <div className="w-2 h-2 rounded-sm mt-1 flex-shrink-0" style={{ background: color }} />
                  <div className="text-xs leading-snug" style={{ color: C.textMid }}>{item}</div>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </LightSlide>
  )
}
