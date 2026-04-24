import { motion, AnimatePresence } from 'framer-motion'
import { DarkSlide } from '../components/SlideLayout'
import { PipelineViz } from '../demo/PipelineViz'
import { TransactionCards } from '../demo/TransactionCards'
import { ProbabilityGauge } from '../demo/ProbabilityGauge'
import { ShapWaterfall } from '../demo/ShapWaterfall'
import { useDemoStore } from '../store/useDemoStore'
import { C } from '../design/tokens'

export function S20_LiveDemo() {
  const { result, stage, reset, error } = useDemoStore()
  const isDone = stage === 'done'

  return (
    <DarkSlide>
      <div className="w-full h-full flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-7 py-4" style={{ background: C.navyMid, borderBottom: `2px solid ${C.teal}` }}>
          <div className="flex items-center gap-3">
            <div className="w-1.5 h-8 rounded" style={{ background: C.teal }} />
            <div>
              <div className="text-xl font-bold text-white">Live Demo</div>
              <div className="text-sm" style={{ color: C.tealBright }}>LightGBM · Fraud Probability + Explainability</div>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm" style={{ color: C.textMuted }}>
            <span>ROC-AUC: <strong style={{ color: C.white }}>0.9191</strong></span>
            <span>PR-AUC: <strong style={{ color: C.white }}>0.5737</strong></span>
          </div>
        </div>

        {/* Main content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Pipeline column */}
          <div className="px-6 py-5 border-r flex-shrink-0 w-56" style={{ borderColor: '#1A3A5C' }}>
            <PipelineViz />
          </div>

          {/* Right panel */}
          <div className="flex-1 flex flex-col p-5 pb-10 gap-2 overflow-hidden">
            {/* Transaction selector */}
            <TransactionCards />

            {error && (
              <div className="text-sm p-3 rounded" style={{ background: '#1A0A0A', color: C.red, border: `1px solid ${C.red}` }}>
                {error}
              </div>
            )}

            {/* Results */}
            <AnimatePresence>
              {isDone && result && (
                <motion.div
                  className="flex gap-5 flex-1 min-h-0 overflow-hidden"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }}
                  exit={{ opacity: 0 }}
                >
                  {/* Probability gauge */}
                  <div
                    className="flex flex-col items-center justify-center rounded-lg p-5 flex-shrink-0"
                    style={{ background: C.navyMid, border: `1px solid #1A3A5C`, minWidth: '190px' }}
                  >
                    <ProbabilityGauge probability={result.probability} />
                  </div>

                  {/* SHAP waterfall */}
                  <div
                    className="flex-1 rounded-lg p-4 overflow-hidden"
                    style={{ background: C.navyMid, border: `1px solid #1A3A5C` }}
                  >
                    <ShapWaterfall shapValues={result.shap_values} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Reset button */}
            {isDone && (
              <motion.button
                className="self-end text-sm px-4 py-2 rounded"
                style={{ background: C.navyMid, color: C.textMuted, border: `1px solid #1A3A5C` }}
                onClick={reset}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1, transition: { delay: 0.5 } }}
                whileHover={{ borderColor: C.teal, color: C.white }}
              >
                Try Another Transaction →
              </motion.button>
            )}
          </div>
        </div>
      </div>
    </DarkSlide>
  )
}
