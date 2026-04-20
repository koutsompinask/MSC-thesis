import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const bullets = [
  { h: 'Transaction Table',  d: 'Contains TransactionDT, TransactionAmt, ProductCD, card types, email domains, and 300+ Vesta-engineered features.' },
  { h: 'Identity Table',     d: 'Device type, browser, OS, network information, and biometric proxy features linked to transactions.' },
  { h: 'Target Variable',    d: 'Binary: isFraud = 1 for fraudulent, 0 for legitimate. Severe imbalance: ~3.5% positive rate.' },
  { h: 'Why IEEE-CIS?',      d: 'Reflects real e-commerce complexity — not a toy dataset. Used in Kaggle competition where top solutions achieved ~96% ROC-AUC.' },
]

const metrics = [
  { val: '590,540', lbl: 'Total Transactions',     bg: C.navyDark },
  { val: '434',     lbl: 'Feature Columns',         bg: C.teal },
  { val: '3.5%',   lbl: 'Fraud Rate (Positive)',   bg: C.red },
  { val: '1:5',    lbl: 'Downsample Ratio Tested', bg: C.purple },
]

export function S06_Dataset() {
  return (
    <LightSlide title="Dataset: IEEE-CIS Fraud Detection" num={6}>
      <div className="flex gap-5 h-full">
        {/* Left */}
        <motion.div
          className="flex-1 rounded overflow-hidden"
          style={{ background: C.bgCard, border: '1px solid #e2e8f0', boxShadow: '0 2px 10px rgba(0,0,0,0.07)' }}
          initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }}
        >
          <div className="px-4 py-2.5 font-bold text-sm text-white" style={{ background: C.navyDark }}>
            Source: IEEE-CIS Kaggle Competition
          </div>
          <div className="px-4 py-3 flex flex-col gap-3">
            {bullets.map(({ h, d }) => (
              <div key={h}>
                <div className="font-semibold text-sm mb-0.5" style={{ color: C.textDark }}>{h}</div>
                <div className="text-xs leading-relaxed" style={{ color: C.textMid }}>{d}</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Right */}
        <motion.div
          className="flex flex-col gap-2.5 w-36"
          initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.15 } }}
        >
          {metrics.map(({ val, lbl, bg }) => (
            <div key={val} className="rounded py-3 text-center flex flex-col items-center gap-0.5" style={{ background: bg }}>
              <div className="font-black text-white text-lg leading-none">{val}</div>
              <div className="text-xs text-white opacity-75 leading-tight">{lbl}</div>
            </div>
          ))}
          <div className="rounded p-2 text-xs text-center leading-snug" style={{ background: '#DBEAFE', border: `1px solid ${C.teal}`, color: C.textDark }}>
            Chronological data split to simulate real deployment
          </div>
        </motion.div>
      </div>
    </LightSlide>
  )
}
