import { motion } from 'framer-motion'
import { useDemoStore } from '../store/useDemoStore'
import type { DemoStage } from '../store/useDemoStore'
import { C } from '../design/tokens'

const stages = [
  { id: 'stage1', label: 'Raw Data',        sub: 'Reading 81 input features' },
  { id: 'stage2', label: 'Preprocessing',   sub: 'Filling missing values → NaN' },
  { id: 'stage3', label: 'Feature Eng.',    sub: 'Behavioral aggregates' },
  { id: 'stage4', label: 'Selection',       sub: 'Top 215 SHAP features' },
  { id: 'stage5', label: 'Model',           sub: 'LightGBM scoring…' },
  { id: 'stage6', label: 'SHAP',            sub: 'Computing attributions' },
]

const stageOrder: DemoStage[] = ['idle', 'selected', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'done']

function stageIndex(s: DemoStage) {
  return stageOrder.indexOf(s)
}

export function PipelineViz() {
  const stage = useDemoStore((s) => s.stage)
  const current = stageIndex(stage)

  return (
    <div className="flex flex-col gap-0 h-full justify-center">
      <div className="text-sm font-bold uppercase tracking-widest mb-4" style={{ color: C.textMuted }}>
        Pipeline
      </div>
      {stages.map(({ id, label, sub }, i) => {
        const stageNum = i + 2 // corresponds to stage1=2, stage2=3...
        const isComplete = current > stageNum
        const isActive = current === stageNum
        const isIdle = current < stageNum

        return (
          <div key={id} className="flex flex-col">
            {/* Connector line */}
            {i > 0 && (
              <div className="ml-5 w-0.5 h-4 overflow-hidden" style={{ background: '#1A3A5C' }}>
                <motion.div
                  className="w-full h-full"
                  style={{ background: C.teal }}
                  initial={{ scaleY: 0 }}
                  animate={{ scaleY: isComplete || isActive ? 1 : 0, transition: { duration: 0.3, ease: 'easeOut' } }}
                />
              </div>
            )}

            {/* Stage row */}
            <motion.div
              className="flex items-center gap-3"
              animate={
                isActive
                  ? { scale: 1.03, transition: { type: 'spring', stiffness: 300, damping: 20 } }
                  : { scale: 1 }
              }
            >
              {/* Circle */}
              <motion.div
                className="w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center font-bold text-sm"
                animate={{
                  background: isComplete ? C.green : isActive ? C.teal : '#1A3A5C',
                  color: isIdle ? '#475569' : '#FFFFFF',
                  transition: { duration: 0.3 }
                }}
              >
                {isComplete ? '✓' : i + 1}
              </motion.div>

              {/* Label */}
              <div>
                <div className="text-sm font-semibold leading-tight"
                  style={{ color: isIdle ? C.textMuted : C.white }}>
                  {label}
                </div>
                {isActive && (
                  <motion.div
                    className="text-xs leading-tight"
                    style={{ color: C.tealBright }}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    {sub}
                  </motion.div>
                )}
              </div>
            </motion.div>
          </div>
        )
      })}
    </div>
  )
}
