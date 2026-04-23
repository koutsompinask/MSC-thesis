import { motion } from 'framer-motion'
import { DarkSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const sections = [
  ['01', 'Challenge & Context',       'The fraud detection problem and why it\'s hard'],
  ['02', 'Literature & Model Choice', 'ML landscape review and gradient boosting rationale'],
  ['03', 'Research Design',           'Four research questions guiding the study'],
  ['04', 'Dataset & Methodology',     'IEEE-CIS data, pipeline, feature engineering, data split'],
  ['05', 'Experimental Results',      'Four configurations across three models'],
  ['06', 'Synthesis & Insights',      'Cross-experiment patterns and what they mean'],
  ['07', 'Live Demo',                 'Live demonstration of the best-performing model pipeline in action'],
  ['08', 'Conclusions',               'Key takeaways and future research directions'],
]

export function S02_Roadmap() {
  return (
    <DarkSlide>
      <div className="w-full h-full flex flex-col px-10 py-8">
        <div className="text-center mb-8">
          <div className="text-sm font-bold tracking-widest uppercase mb-2" style={{ color: C.teal }}>Presentation Roadmap</div>
          <div className="text-4xl font-bold text-white">What We'll Cover Today</div>
        </div>

        <div
          className="flex-1 grid grid-cols-2 gap-4"
          style={{ gridTemplateRows: 'repeat(4, minmax(0, 1fr))' }}
        >
          {sections.map(([num, title, desc], i) => (
            <motion.div
              key={num}
              className="flex overflow-hidden rounded h-full"
              style={{ background: C.navy, border: `1px solid ${C.navyMid}` }}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0, transition: { delay: i * 0.07, duration: 0.35 } }}
            >
              <div className="w-16 flex-shrink-0 flex items-center justify-center font-black text-xl text-white" style={{ background: C.teal }}>
                {num}
              </div>
              <div className="flex-1 py-4 px-5 flex flex-col justify-center">
                <div className="font-bold text-lg text-white leading-tight">{title}</div>
                <div className="text-base mt-1 leading-snug" style={{ color: C.textMuted }}>{desc}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </DarkSlide>
  )
}
