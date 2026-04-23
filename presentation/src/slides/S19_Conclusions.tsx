import { motion } from 'framer-motion'
import { LightSlide } from '../components/SlideLayout'
import { AccentCard } from '../components/AccentCard'
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
      <div className="flex gap-5 h-full">
        {/* Left: conclusions */}
        <motion.div className="flex-1 rounded overflow-hidden"
          style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
          initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }}>
          <div className="px-4 py-2.5 font-bold text-base text-white" style={{ background: C.navyDark }}>KEY CONCLUSIONS</div>
          <div className="px-4 py-3 flex flex-col gap-3">
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
        <motion.div className="w-56 rounded overflow-hidden"
          style={{ background: C.bgCard, border: '1px solid #e2e8f0' }}
          initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.15 } }}>
          <div className="px-4 py-2.5 font-bold text-base text-white" style={{ background: C.navyDark }}>FUTURE RESEARCH</div>
          <div className="px-3 py-3 flex flex-col gap-2">
            {future.map(({ color, title, desc }, i) => (
              <motion.div key={title} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0, transition: { delay: 0.2 + i * 0.09 } }}>
                <AccentCard accent={color} title={title}>
                  <div className="text-sm leading-relaxed" style={{ color: C.textMid }}>{desc}</div>
                </AccentCard>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </LightSlide>
  )
}
