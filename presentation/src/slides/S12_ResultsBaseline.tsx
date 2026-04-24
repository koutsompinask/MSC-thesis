import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell, LabelList } from 'recharts'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const chartData = [
  { model: 'XGBoost',  auc: 0.896, color: C.amber  },
  { model: 'LightGBM', auc: 0.918, color: C.teal   },
  { model: 'CatBoost', auc: 0.910, color: C.purple  },
]

const tableRows = [
  { model: 'XGBoost',  mc: C.amber,  auc: '0.896', prauc: '0.496', f1: '0.384', prec: '0.269', rec: '0.671' },
  { model: 'LightGBM', mc: C.teal,   auc: '0.918', prauc: '0.537', f1: '0.506', prec: '0.588', rec: '0.456' },
  { model: 'CatBoost', mc: C.purple, auc: '0.910', prauc: '0.510', f1: '0.475', prec: '0.490', rec: '0.461' },
]

export function S12_ResultsBaseline() {
  return (
    <LightSlide title="Results: Baseline — Full Features, Imbalanced Data" num={12}>
      <div className="flex flex-col h-full gap-2">
        <motion.div className="px-4 py-2 text-sm font-bold text-white rounded"
          style={{ background: C.teal }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          LightGBM achieves the strongest baseline ROC-AUC at 0.918, ahead of CatBoost (0.910) and XGBoost (0.896)
        </motion.div>

        <div className="flex gap-4 flex-1 min-h-0 pb-7">
          {/* Chart */}
          <motion.div className="flex-1" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.15 } }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="model" tick={{ fill: C.textMid, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0.87, 0.93]} tick={{ fill: C.textMid, fontSize: 10 }} axisLine={false} tickLine={false} tickCount={4} />
                <Bar dataKey="auc" radius={[4, 4, 0, 0]}>
                  {chartData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  <LabelList dataKey="auc" position="top" style={{ fill: C.textDark, fontSize: 11, fontWeight: 700 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Table */}
          <motion.div className="w-96 flex flex-col" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.2 } }}>
            <div className="rounded overflow-hidden" style={{ border: '1px solid #e2e8f0' }}>
              <div
                className="grid text-xs font-bold text-white py-1.5"
                style={{ background: C.navyDark, gridTemplateColumns: '92px repeat(5, minmax(0, 1fr))' }}
              >
                {['Model', 'AUC', 'PR-AUC', 'F1', 'Prec.', 'Rec.'].map(h => (
                  <div key={h} className={h === 'Model' ? 'pl-4' : 'text-center'}>{h}</div>
                ))}
              </div>
              {tableRows.map(({ model, mc, auc, prauc, f1, prec, rec }, ri) => (
                <div
                  key={model}
                  className="grid text-xs py-2 border-t"
                  style={{ background: ri % 2 === 0 ? C.bgCard : '#F1F7FF', borderColor: '#e2e8f0', gridTemplateColumns: '92px repeat(5, minmax(0, 1fr))' }}
                >
                  <div className="flex items-center pl-2 pr-1 gap-1 min-w-0">
                    <div className="w-1 self-stretch" style={{ background: mc }} />
                    <span className="font-bold ml-1 truncate" style={{ color: mc }}>{model}</span>
                  </div>
                  {[auc, prauc, f1, prec, rec].map((v, i) => (
                    <div key={i} className="text-center font-medium" style={{ color: C.textDark }}>{v}</div>
                  ))}
                </div>
              ))}
            </div>
            <div className="mt-2 rounded p-2.5 text-sm leading-snug" style={{ background: '#DBEAFE', border: `1px solid ${C.teal}`, color: C.textDark }}>
              <strong>XGBoost</strong> shows highest recall (0.671) but much lower precision (0.269). <strong>LightGBM</strong> achieves the best overall balance: highest AUC, PR-AUC, and F1.
            </div>
          </motion.div>
        </div>
      </div>
    </LightSlide>
  )
}
