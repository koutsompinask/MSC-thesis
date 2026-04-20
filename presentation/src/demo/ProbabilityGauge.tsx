import { motion } from 'framer-motion'
import { C } from '../design/tokens'

interface Props { probability: number }

export function ProbabilityGauge({ probability }: Props) {
  const pct = Math.round(probability * 100)
  // SVG semi-circle gauge
  const r = 52
  const cx = 70
  const cy = 70
  const circumference = Math.PI * r  // half circle
  const dash = circumference * probability
  const gap  = circumference * (1 - probability)

  // Color: teal for low, amber for mid, high stays amber/orange (no red FRAUD label)
  const gaugeColor = probability < 0.2 ? C.teal : probability < 0.5 ? C.tealBright : C.amber

  return (
    <div className="flex flex-col items-center gap-2">
      {/* SVG gauge */}
      <svg width="140" height="85" viewBox="0 0 140 85">
        {/* Track */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="#1A3A5C"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Animated fill */}
        <motion.path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={gaugeColor}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${gap + 1}`}
          initial={{ strokeDasharray: `0 ${circumference + 1}` }}
          animate={{ strokeDasharray: `${dash} ${gap + 1}`, transition: { duration: 0.9, ease: 'easeOut' } }}
        />
        {/* Center value */}
        <text x={cx} y={cy - 4} textAnchor="middle" fontWeight="800" fontSize="22" fill="white">
          {pct}%
        </text>
        {/* Label */}
        <text x={cx} y={cy + 12} textAnchor="middle" fontSize="9" fill="#94A3B8" fontWeight="500">
          Fraud Probability
        </text>
        {/* 0% / 100% markers */}
        <text x={cx - r - 4} y={cy + 16} textAnchor="middle" fontSize="7" fill="#475569">0%</text>
        <text x={cx + r + 4} y={cy + 16} textAnchor="middle" fontSize="7" fill="#475569">100%</text>
      </svg>

      {/* Probability pill */}
      <motion.div
        className="px-4 py-1.5 rounded-full text-xs font-bold text-center"
        style={{ background: `${gaugeColor}22`, border: `1px solid ${gaugeColor}`, color: gaugeColor }}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1, transition: { delay: 0.5 } }}
      >
        {pct}% fraud probability
      </motion.div>

      <div className="text-xs text-center" style={{ color: C.textMuted, maxWidth: '140px', lineHeight: 1.4 }}>
        Raw model score — threshold selection determines classification
      </div>
    </div>
  )
}
