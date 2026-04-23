import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { AccentCard } from '../components/AccentCard'
import { C } from '../design/tokens'

const challenges = [
  { color: C.red,    title: 'Extreme Class Imbalance',        desc: 'Fraudulent transactions represent only ~3.5% of all activity. Standard models learn to predict the majority class and miss fraud entirely.' },
  { color: C.amber,  title: 'Continuously Evolving Tactics',  desc: 'Fraudsters constantly adapt their methods to evade detection, making rule-based systems obsolete within months.' },
  { color: C.green,  title: 'False Positives vs. False Negatives', desc: 'Catching more fraud usually means flagging more legitimate transactions. The operating threshold must balance fraud losses against analyst workload and customer friction.' },
  { color: C.teal,   title: 'Real-Time Constraints',          desc: 'Detection decisions must be made in milliseconds at scale — complex models must also be computationally practical.' },
  { color: C.purple, title: 'Interpretability vs. Accuracy',  desc: 'Complex ensemble models achieve the best results, but financial regulators and risk teams demand explainable decisions.' },
]


export function S03_Challenge() {
  return (
    <LightSlide title="The Fraud Detection Challenge" num={3}>
      <div className="flex gap-4 h-full">
        {/* Left: challenges */}
        <div
          className="grid flex-1 gap-2.5"
          style={{ gridTemplateRows: `repeat(${challenges.length}, minmax(0, 1fr))` }}
        >
          {challenges.map((c, i) => (
            <motion.div
              key={i}
              className="h-full"
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0, transition: { delay: i * 0.08 } }}
            >
              <AccentCard accent={c.color} className="h-full">
                <div className="font-semibold text-base mb-1.5 leading-snug" style={{ color: C.textDark }}>
                  {c.title}
                </div>
                <div className="text-sm leading-relaxed" style={{ color: C.textMid }}>
                  {c.desc}
                </div>
              </AccentCard>
            </motion.div>
          ))}
        </div>

        {/* Right: stats */}
        <motion.div
          className="flex flex-col gap-2 w-36 pb-6"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0, transition: { delay: 0.3 } }}
        >
          {/* Big fraud rate stat */}
          <div
            className="flex-1 flex flex-col items-center justify-center rounded text-center px-2"
            style={{ background: C.navyDark, border: `1.5px solid ${C.teal}`, minHeight: '0' }}
          >
            <div className="text-4xl font-black leading-none" style={{ color: C.tealBright }}>3.5%</div>
            <div className="text-xs mt-1 leading-snug" style={{ color: '#94A3B8' }}>
              fraud rate
              <br />
              in IEEE-CIS
            </div>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {[
              { val: '590K+', lbl: 'Transactions', bg: C.teal },
              { val: '434',   lbl: 'Features',      bg: C.navyMid },
              { val: '3',     lbl: 'ML Models',     bg: C.purple },
              { val: '4',     lbl: 'RQs',           bg: C.amber },
            ].map(({ val, lbl, bg }) => (
              <div key={val} className="rounded py-2 text-center" style={{ background: bg }}>
                <div className="font-bold text-white text-lg leading-none">{val}</div>
                <div className="text-white text-xs opacity-80 mt-0.5 leading-tight">{lbl}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </LightSlide>
  )
}
