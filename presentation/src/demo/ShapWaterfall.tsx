import { motion } from 'framer-motion'
import { useState } from 'react'
import type { ShapEntry } from '../store/useDemoStore'
import { C } from '../design/tokens'

interface Props {
  shapValues: ShapEntry[]
  baseValue: number
}

export function ShapWaterfall({ shapValues, baseValue }: Props) {
  const [tooltip, setTooltip] = useState<number | null>(null)

  if (!shapValues.length) return null

  const maxAbs = Math.max(...shapValues.map((s) => Math.abs(s.shap)))
  const formatFeatureValue = (value: ShapEntry['value']) => {
    if (typeof value === 'number') return value.toFixed(3)
    if (value == null || value === '') return 'missing'
    return String(value)
  }

  return (
    <div className="flex flex-col h-full">
      <div className="text-xs font-bold mb-2" style={{ color: C.white }}>What drove this score?</div>

      <div className="flex flex-col gap-1 flex-1">
        {shapValues.slice(0, 10).map((entry, i) => {
          const barWidth = Math.abs(entry.shap) / maxAbs * 100
          const isPos = entry.shap > 0
          const barColor = isPos ? C.amber : C.teal

          return (
            <div
              key={i}
              className="flex items-center gap-1.5 cursor-default relative"
              onMouseEnter={() => setTooltip(i)}
              onMouseLeave={() => setTooltip(null)}
            >
              {/* Feature name */}
              <div
                className="text-xs text-right flex-shrink-0 truncate"
                style={{ color: C.textMuted, width: '110px', fontSize: '10px' }}
                title={entry.feature}
              >
                {entry.feature.length > 16 ? entry.feature.slice(0, 15) + '…' : entry.feature}
              </div>

              {/* Bar track (centered) */}
              <div className="flex-1 h-5 relative flex items-center">
                {/* Center divider */}
                <div className="absolute left-1/2 top-0 h-full w-px" style={{ background: '#1A3A5C' }} />

                {isPos ? (
                  <motion.div
                    className="absolute"
                    style={{ left: '50%', height: '100%', borderRadius: '0 3px 3px 0', background: barColor, maxWidth: '50%' }}
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth / 2}%`, transition: { delay: 0.1 + i * 0.05, duration: 0.4, ease: 'easeOut' } }}
                  />
                ) : (
                  <motion.div
                    className="absolute"
                    style={{ right: '50%', height: '100%', borderRadius: '3px 0 0 3px', background: barColor, maxWidth: '50%' }}
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth / 2}%`, transition: { delay: 0.1 + i * 0.05, duration: 0.4, ease: 'easeOut' } }}
                  />
                )}
              </div>

              {/* SHAP value */}
              <div
                className="text-xs font-bold flex-shrink-0 w-10 text-right"
                style={{ color: barColor, fontSize: '10px' }}
              >
                {entry.shap > 0 ? '+' : ''}{entry.shap.toFixed(2)}
              </div>

              {/* Tooltip */}
              {tooltip === i && (
                <motion.div
                  className="absolute z-50 px-2.5 py-2 rounded text-xs pointer-events-none"
                  style={{ background: C.navyDark, border: `1px solid ${C.navyMid}`, color: C.white, top: '-40px', left: '110px', minWidth: '180px', boxShadow: '0 4px 12px rgba(0,0,0,0.4)' }}
                  initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                >
                  <div className="font-bold" style={{ color: barColor }}>{entry.feature}</div>
                  <div style={{ color: C.textMuted }}>value: {formatFeatureValue(entry.value)}</div>
                  <div>SHAP: <span style={{ color: barColor }}>{entry.shap > 0 ? '+' : ''}{entry.shap.toFixed(4)}</span></div>
                </motion.div>
              )}
            </div>
          )
        })}
      </div>

      <div className="text-xs mt-1.5 pt-1.5" style={{ color: C.textMuted, borderTop: `1px solid ${C.navyMid}`, fontSize: '9px' }}>
        SHAP values (log-odds). Positive = increased fraud probability.
        Base value: {baseValue.toFixed(3)}
      </div>
    </div>
  )
}
