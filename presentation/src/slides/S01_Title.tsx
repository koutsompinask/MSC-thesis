import { motion } from 'framer-motion'
import type { Variants } from 'framer-motion'
import { DarkSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const stagger: Variants = { hidden: {}, show: { transition: { staggerChildren: 0.1 } } }
const item: Variants = { hidden: { opacity: 0, x: -16 }, show: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 200, damping: 22 } } }

export function S01_Title() {
  return (
    <DarkSlide>
      <div className="relative w-full h-full flex overflow-hidden">
        {/* Left teal stripe */}
        <div className="w-3 flex-shrink-0" style={{ background: C.teal }} />

        {/* Subtle horizontal lines */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="absolute w-full h-px" style={{ top: `${i * 12.5}%`, background: C.teal, opacity: 0.07 }} />
          ))}
        </div>

        {/* Main content */}
        <motion.div className="flex-1 flex flex-col justify-center pl-10 pr-4" variants={stagger} initial="hidden" animate="show">
          {/* Chip */}
          <motion.div variants={item} className="inline-flex mb-4">
            <div className="px-3 py-1 text-xs font-bold text-white tracking-widest uppercase" style={{ background: C.teal }}>
              MSc Thesis Presentation
            </div>
          </motion.div>

          {/* Title */}
          <motion.h1 variants={item} className="font-black leading-none text-white mb-3" style={{ fontSize: 'clamp(2.5rem, 5vw, 4rem)', letterSpacing: '-0.02em' }}>
            BEHAVIORAL<br />FRAUD ANALYTICS
          </motion.h1>

          {/* Subtitle */}
          <motion.p variants={item} className="text-sm font-medium mb-8 leading-relaxed" style={{ color: C.tealBright, maxWidth: '600px' }}>
            Machine Learning for Fraud Detection in Online Financial Transactions
          </motion.p>

          {/* Divider */}
          <motion.div variants={item} className="h-px w-3/4 mb-5" style={{ background: C.teal }} />

          {/* Author */}
          <motion.div variants={item} className="text-xs leading-relaxed" style={{ color: C.textMuted }}>
            <span className="font-bold text-white">Koutsompinas Konstantinos</span><br />
            Supervisor: Athanasios Argyriou<br />
            National &amp; Kapodistrian University of Athens · Dept. of Economics · February 2026
          </motion.div>
        </motion.div>

        {/* Right stats */}
        <motion.div
          className="flex flex-col justify-center gap-4 pr-10"
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0, transition: { delay: 0.5, duration: 0.5 } }}
        >
          <div className="px-5 py-4 text-center rounded" style={{ background: C.navyMid, border: `1.5px solid ${C.teal}`, minWidth: '130px' }}>
            <div className="text-4xl font-black leading-none" style={{ color: C.tealBright }}>3.5%</div>
            <div className="text-xs mt-1" style={{ color: C.textMuted }}>fraud rate<br />IEEE-CIS dataset</div>
          </div>
          <div className="px-5 py-4 text-center rounded" style={{ background: C.navyMid, border: `1px solid ${C.navyMid}`, minWidth: '130px' }}>
            <div className="text-3xl font-black leading-none text-white">590K+</div>
            <div className="text-xs mt-1" style={{ color: C.textMuted }}>transactions<br />analyzed</div>
          </div>
        </motion.div>
      </div>
    </DarkSlide>
  )
}
