import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { C } from '../design/tokens'

const conclusions = [
  { n: 1, text: 'Gradient boosting models are highly effective for fraud detection on the IEEE-CIS dataset, achieving strong discrimination without deep learning complexity.' },
  { n: 2, text: 'LightGBM was the most dependable model overall, with ROC-AUC ≈ 0.918–0.919 across configurations — making it the recommended baseline choice.' },
  { n: 3, text: 'Behavioral feature engineering materially improved model insight and performance. Engineered features consistently ranked among top predictors via SHAP analysis.' },
  { n: 4, text: 'SHAP explainability was implemented for all models, delivering both global feature rankings and per-prediction local attribution — bridging accuracy and interpretability.' },
]

const future = [
  { color: C.purple, title: 'Advanced Imbalance Methods',   desc: 'Cost-sensitive learning, focal losses, SMOTE variants on harder datasets' },
  { color: C.teal,   title: 'Real-Time SHAP Deployment',    desc: 'Embed SHAP scoring in live inference pipelines for regulatory compliance' },
  { color: C.green,  title: 'Ensemble & Hybrid Systems',    desc: 'Combine rule-based and ML approaches for broader fraud coverage' },
  { color: C.amber,  title: 'Model Drift Monitoring',       desc: 'Longitudinal monitoring of model performance as fraud tactics evolve' },
]

export function S19_Conclusions() {
  return (
    <LightSlide title="Conclusions & Future Research" num={19}>
      <div className="grid grid-cols-[3fr_2fr] gap-4 h-full pb-7">
        {/* Left: conclusions */}
        <motion.div className="rounded overflow-hidden min-w-0"
          style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
          initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }}>
          <div className="px-4 py-2.5 font-bold text-base text-white" style={{ background: C.navyDark }}>KEY CONCLUSIONS</div>
          <div className="px-4 py-3 flex flex-col gap-2.5">
            {conclusions.map(({ n, text }, i) => (
              <motion.div key={n} className="flex items-start gap-3"
                initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.1 + i * 0.1 } }}>
                <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center font-bold text-sm text-white" style={{ background: C.teal }}>{n}</div>
                <div className="text-sm leading-relaxed pt-0.5" style={{ color: C.textMid }}>{text}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Right: future work */}
        <motion.div className="rounded overflow-hidden min-w-0"
          style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
          initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.15 } }}>
          <div className="px-4 py-2.5 font-bold text-base text-white" style={{ background: C.navyDark }}>FUTURE RESEARCH</div>
          <div className="px-3 py-3 grid grid-rows-4 gap-2 h-[calc(100%-42px)]">
            {future.map(({ color, title, desc }, i) => (
              <motion.div
                key={title}
                className="flex rounded overflow-hidden"
                style={{ background: '#F8FBFF', border: '1px solid #e2e8f0' }}
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0, transition: { delay: 0.2 + i * 0.09 } }}
              >
                <div className="w-1.5 flex-shrink-0" style={{ background: color }} />
                <div className="px-3 py-2 flex flex-col justify-center">
                  <div className="font-semibold text-sm leading-tight mb-1" style={{ color: C.textDark }}>{title}</div>
                  <div className="text-xs leading-snug" style={{ color: C.textMid }}>{desc}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </LightSlide>
  )
}
