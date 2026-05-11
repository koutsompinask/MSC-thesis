import { motion } from 'framer-motion'
import { DarkSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const tableData = [
  { model: 'LightGBM', mc: C.teal,   vals: ['0.918', '0.917', '0.919'], best: [false, false, true] },
  { model: 'CatBoost', mc: C.purple, vals: ['0.910', '0.916', '0.914'], best: [false, false, false] },
  { model: 'XGBoost',  mc: C.amber,  vals: ['0.896', '0.908', '0.905'], best: [false, false, false] },
]

const colHeaders = ['Model', 'Baseline\n(Imbalanced)', 'Config 2\n(Downsampled)', 'Config 3\n(Reduced\nFeatures)']

export function S17_ChampionScorecard() {
  return (
    <DarkSlide>
      <div className="w-full h-full flex flex-col px-10 py-6">
        <div className="text-center mb-5">
          <div className="text-sm font-bold tracking-widest uppercase mb-1" style={{ color: C.teal }}>Champion Scorecard</div>
          <div className="text-2xl font-bold text-white">ROC-AUC Across All Configurations</div>
        </div>

        <motion.div
          className="flex-1 rounded overflow-hidden"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0, transition: { duration: 0.4 } }}
        >
          {/* Header */}
          <div className="grid" style={{ gridTemplateColumns: '180px repeat(3, 1fr)' }}>
            {colHeaders.map((h, i) => (
              <div key={i} className="flex items-center justify-center text-center font-bold text-sm text-white py-3 px-2"
                style={{ background: i === 0 ? C.navyMid : C.teal, borderRight: '1px solid rgba(255,255,255,0.1)', whiteSpace: 'pre-line', lineHeight: 1.3 }}>
                {h}
              </div>
            ))}
          </div>

          {/* Rows */}
          {tableData.map(({ model, mc, vals, best }, ri) => (
            <motion.div
              key={model}
              className="grid"
              style={{ gridTemplateColumns: '180px repeat(3, 1fr)' }}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0, transition: { delay: 0.15 + ri * 0.1 } }}
            >
              <div className="flex items-center py-5 px-4" style={{ background: C.navy, borderRight: '1px solid rgba(255,255,255,0.06)', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <div className="w-1.5 self-stretch mr-3" style={{ background: mc }} />
                <span className="font-bold text-lg" style={{ color: mc }}>{model}</span>
              </div>
              {vals.map((v, vi) => (
                <div key={vi} className="flex items-center justify-center font-black text-2xl py-5"
                  style={{
                    background: best[vi] ? C.teal : (ri === 0 ? C.navyMid : C.navy),
                    color: ri === 0 ? C.white : C.tealPale,
                    borderTop: '1px solid rgba(255,255,255,0.06)',
                    borderRight: '1px solid rgba(255,255,255,0.06)',
                  }}>
                  {v}
                </div>
              ))}
            </motion.div>
          ))}
        </motion.div>

        <motion.div className="mt-4 rounded px-4 py-2.5 text-sm leading-relaxed" style={{ background: C.navyMid, border: `1px solid ${C.teal}`, color: C.tealBright }}
          initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.5 } }}>
          LightGBM provides the most stable ROC-AUC performance. Thesis rationale: histogram-based split finding handles the high-dimensional feature space efficiently and may regularize under severe imbalance.
        </motion.div>
      </div>
    </DarkSlide>
  )
}
