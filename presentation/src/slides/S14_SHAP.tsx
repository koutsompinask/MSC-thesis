import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const features = [
  { name: 'C1',                           val: 0.422, color: C.navyMid },
  { name: 'C13',                          val: 0.239, color: C.navyMid },
  { name: 'V70',                          val: 0.180, color: '#06B6D4' },
  { name: 'uid_V258_mean',                val: 0.143, color: C.teal },
  { name: 'ProductCD_TransactionAmt_rel', val: 0.141, color: C.green },
  { name: 'uid_C1_mean',                  val: 0.136, color: C.teal },
  { name: 'DeviceInfo',                   val: 0.132, color: C.purple },
  { name: 'V258',                         val: 0.131, color: '#06B6D4' },
  { name: 'uid_V283_mean',                val: 0.130, color: C.teal },
  { name: 'uid_C13_std',                  val: 0.130, color: C.teal },
  { name: 'uid_C13_rel',                  val: 0.129, color: C.teal },
  { name: 'card1_id_02_mean',             val: 0.114, color: C.green },
  { name: 'P_emaildomain',                 val: 0.107, color: C.purple },
  { name: 'uid_C13_mean',                  val: 0.104, color: C.teal },
  { name: 'uid_D2_std',                    val: 0.101, color: C.teal },
]

export function S14_SHAP() {
  return (
    <LightSlide title="SHAP Feature Importance: Selection Rationale" num={14}>
      <div className="flex flex-col h-full gap-2">
        <div className="flex gap-4 flex-1 min-h-0">
          <div className="flex-1 rounded overflow-hidden" style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}>
            <div className="px-3 py-2 font-bold text-sm text-white" style={{ background: C.navyDark }}>
              Table 4: Mean |SHAP| Importance Across XGBoost, LightGBM & CatBoost
            </div>
            <div className="px-3 py-2 flex flex-col gap-1">
              {features.map(({ name, val, color }, i) => (
                <div key={name} className="flex items-center gap-2">
                  <div className="w-4 text-xs font-bold text-right flex-shrink-0" style={{ color: C.textMuted }}>
                    {i + 1}
                  </div>
                  <div className="flex-1 relative h-5 rounded overflow-hidden" style={{ background: '#DDE8F4' }}>
                    <motion.div
                      className="absolute left-0 top-0 h-full flex items-center pl-2 pr-2"
                      style={{ background: color, width: `${(val / features[0].val) * 100}%` }}
                      initial={{ width: 0 }}
                      animate={{ width: `${(val / features[0].val) * 100}%`, transition: { delay: 0.2 + i * 0.04, duration: 0.45, ease: 'easeOut' } }}
                    >
                      <span className="text-white font-bold text-xs truncate">{name}</span>
                    </motion.div>
                  </div>
                  <div className="text-xs font-bold w-10 text-right flex-shrink-0" style={{ color: C.textDark }}>
                    {val.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="w-56 flex flex-col gap-2">
            {[
              { color: C.teal, title: 'Global Selection',
                body: 'Mean absolute SHAP ranks the variables driving model predictions across the trained gradient boosting models.' },
              { color: C.green, title: 'Cross-Model Agreement',
                body: 'Features in the top 30% for at least two models were kept, reducing 748 model inputs to 215.' },
              { color: C.purple, title: 'Local Explanations',
                body: 'The same SHAP machinery supports per-transaction attribution in the live demo.' },
            ].map(({ color, title, body }, i) => (
              <motion.div
                key={title}
                className="rounded overflow-hidden flex-1"
                style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0, transition: { delay: 0.2 + i * 0.08 } }}
              >
                <div className="px-3 py-1.5 font-bold text-sm text-white" style={{ background: color }}>
                  {title}
                </div>
                <div className="px-3 py-2 text-sm leading-snug" style={{ color: C.textMid }}>
                  {body}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <motion.div
          className="px-4 py-2 text-sm font-bold text-white rounded"
          style={{ background: C.teal }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1, transition: { delay: 0.6 } }}
        >
          These SHAP rankings justify the reduced-feature experiment: engineered behavioral aggregates appear alongside original C, V and identity/device fields.
        </motion.div>
      </div>
    </LightSlide>
  )
}
