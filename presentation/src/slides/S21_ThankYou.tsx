import { motion } from 'framer-motion'
import { DarkSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

export function S21_ThankYou() {
  return (
    <DarkSlide>
      <div className="relative w-full h-full flex overflow-hidden">
        {/* Left teal stripe */}
        <div className="w-3 flex-shrink-0" style={{ background: C.teal }} />

        <motion.div className="flex-1 flex flex-col justify-center pl-10 pr-16"
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { duration: 0.5 } }}>
          <motion.h1 className="font-black text-white leading-none mb-4"
            style={{ fontSize: 'clamp(3rem, 7vw, 5rem)', letterSpacing: '-0.02em' }}
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.1 } }}>
            THANK YOU
          </motion.h1>
          <motion.p className="text-xl font-light mb-8" style={{ color: C.tealBright }}
            initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.2 } }}>
            Thank you for your attention
          </motion.p>
          <motion.div
            style={{ background: C.teal, height: '1.5px', transformOrigin: 'left', width: '100%', marginBottom: '1.5rem' }}
            initial={{ scaleX: 0 }} animate={{ scaleX: 1, transition: { delay: 0.3, duration: 0.5 } }}
          />
          <motion.div className="text-xs leading-relaxed" style={{ color: '#94A3B8' }}
            initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.4 } }}>
            <span className="font-bold text-white">Konstantinos Koutsompinas</span><br />
            Supervisor: Athanasios Argyriou<br />
            National &amp; Kapodistrian University of Athens
          </motion.div>
        </motion.div>
      </div>
    </DarkSlide>
  )
}
