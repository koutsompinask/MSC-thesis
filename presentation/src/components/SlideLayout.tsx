import { motion } from 'framer-motion'
import type { ReactNode } from 'react'
import { C } from '../design/tokens'

const slideVariants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.35 } },
  exit:    { opacity: 0, y: -8, transition: { duration: 0.2 } },
}

interface DarkSlideProps { children: ReactNode; className?: string }
interface LightSlideProps { title: string; num?: number; children: ReactNode; accent?: string }

export function DarkSlide({ children, className = '' }: DarkSlideProps) {
  return (
    <motion.div
      variants={slideVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className={`w-full h-full flex flex-col overflow-hidden ${className}`}
      style={{ background: C.navyDark }}
    >
      {children}
    </motion.div>
  )
}

export function LightSlide({ title, num, children, accent = C.teal }: LightSlideProps) {
  return (
    <motion.div
      variants={slideVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="w-full h-full flex flex-col overflow-hidden"
      style={{ background: C.bgPage }}
    >
      {/* Top accent line */}
      <div className="h-1 w-full" style={{ background: accent }} />

      {/* Header band */}
      <div
        className="flex items-center justify-between px-8 py-3"
        style={{ background: C.navyDark, minHeight: '52px' }}
      >
        <h2
          className="font-bold tracking-wide uppercase text-sm"
          style={{ color: C.tealBright, letterSpacing: '0.08em' }}
        >
          {title}
        </h2>
        {num != null && (
          <span className="text-xs font-medium" style={{ color: C.textMuted }}>
            {num}
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden p-6">
        {children}
      </div>
    </motion.div>
  )
}
