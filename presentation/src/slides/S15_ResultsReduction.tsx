import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const chartData = [
  { model: 'XGBoost',  full: 0.908, reduced: 0.905 },
  { model: 'LightGBM', full: 0.917, reduced: 0.919 },
  { model: 'CatBoost', full: 0.916, reduced: 0.914 },
]

const insights = [
  { color: C.green,  title: 'Performance Maintained',  desc: 'All three models retain near-identical ROC-AUC after compression. LightGBM marginally improves to 0.919.' },
  { color: C.teal,   title: 'Top Engineered Features', desc: 'Behavioral aggregates (_mean, _rel, _avg, _std, _freq) consistently rank as the strongest predictors.' },
  { color: C.purple, title: 'SHAP-Driven Selection',   desc: 'Top 30% importance per model, kept when ≥2 models agree — reduces model inputs from 748 to 215.' },
  { color: C.amber,  title: 'Deployment Advantage',    desc: 'Fewer features = faster inference, simpler pipelines, lower maintenance cost in production.' },
]

export function S15_ResultsReduction() {
  return (
    <LightSlide title="Results: Feature Set Reduction" num={15}>
      <div className="flex flex-col h-full gap-2">
        <motion.div className="px-4 py-2 text-sm font-bold text-white rounded" style={{ background: C.green }}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          Compact models remain highly competitive — LightGBM slightly improves to 0.919 ROC-AUC with reduced features
        </motion.div>

        <div className="flex gap-4 flex-1 min-h-0 pb-7">
          <motion.div className="flex-1 min-w-0" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.15 } }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 18, right: 10, bottom: 16, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="model" tick={{ fill: C.textMid, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0.87, 0.93]} tick={{ fill: C.textMid, fontSize: 10 }} axisLine={false} tickLine={false} tickCount={4} />
                <Legend wrapperStyle={{ fontSize: 11, color: C.textMid }} />
                <Bar dataKey="full" name="Downsampled Full Features" fill={C.textMuted} radius={[4, 4, 0, 0]} />
                <Bar dataKey="reduced" name="Reduced Features" fill={C.teal} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          <div className="w-64 grid grid-rows-4 gap-2">
            {insights.map(({ color, title, desc }, i) => (
              <motion.div
                key={title}
                className="flex rounded overflow-hidden"
                style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 8px rgba(0,0,0,0.07)' }}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0, transition: { delay: 0.1 + i * 0.08 } }}
              >
                <div className="w-1.5 flex-shrink-0" style={{ background: color }} />
                <div className="flex-1 px-3 py-2">
                  <div className="font-semibold text-sm leading-tight mb-1" style={{ color: C.textDark }}>{title}</div>
                  <div className="text-xs leading-snug" style={{ color: C.textMid }}>{desc}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </LightSlide>
  )
}
