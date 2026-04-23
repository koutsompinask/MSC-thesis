import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'
import { LightSlide } from '../components/SlideLayout'
import { AccentCard } from '../components/AccentCard'
import { C } from '../design/tokens'

const chartData = [
  { model: 'XGBoost',  full: 0.896, reduced: 0.905 },
  { model: 'LightGBM', full: 0.918, reduced: 0.919 },
  { model: 'CatBoost', full: 0.910, reduced: 0.914 },
]

const insights = [
  { color: C.green,  title: 'Performance Maintained',  desc: 'All three models retain near-identical ROC-AUC after compression. LightGBM marginally improves to 0.919.' },
  { color: C.teal,   title: 'Top Engineered Features', desc: 'Behavioral aggregates (_mean, _rel, _avg, _std, _freq) consistently rank as the strongest predictors.' },
  { color: C.purple, title: 'SHAP-Driven Selection',   desc: 'Top 30% importance per model, kept when ≥2 models agree — reduces feature set from 434 to 215.' },
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

        <div className="flex gap-5 flex-1">
          <motion.div className="flex-1" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.15 } }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="model" tick={{ fill: C.textMid, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0.87, 0.93]} tick={{ fill: C.textMid, fontSize: 10 }} axisLine={false} tickLine={false} tickCount={4} />
                <Legend wrapperStyle={{ fontSize: 11, color: C.textMid }} />
                <Bar dataKey="full" name="Full Features" fill={C.textMuted} radius={[4, 4, 0, 0]} />
                <Bar dataKey="reduced" name="Reduced Features" fill={C.teal} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          <div className="w-52 flex flex-col gap-2">
            {insights.map(({ color, title, desc }, i) => (
              <motion.div key={title} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.1 + i * 0.08 } }}>
                <AccentCard accent={color} title={title}>
                  <div className="text-sm leading-relaxed" style={{ color: C.textMid }}>{desc}</div>
                </AccentCard>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </LightSlide>
  )
}
