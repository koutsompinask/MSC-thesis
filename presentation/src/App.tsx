import { useEffect, useRef } from 'react'
import { AnimatePresence } from 'framer-motion'
import { usePresentationStore } from './store/usePresentationStore'
import { useKeyNav } from './hooks/useKeyNav'
import { NavBar } from './components/NavBar'
import { PointerOverlay } from './remote/PointerOverlay'
import { RemoteControlPage } from './remote/RemoteControlPage'
import { usePresenterRemote } from './remote/usePresenterRemote'
import { S01_Title } from './slides/S01_Title'
import { S02_Roadmap } from './slides/S02_Roadmap'
import { S03_Challenge } from './slides/S03_Challenge'
import { S04_ResearchQuestions } from './slides/S04_ResearchQuestions'
import { S05_Literature } from './slides/S05_Literature'
import { S06_Dataset } from './slides/S06_Dataset'
import { S07_Methodology } from './slides/S07_Methodology'
import { S08_DataSplit } from './slides/S08_DataSplit'
import { S09_EDA } from './slides/S09_EDA'
import { S10_FeatureEngineering } from './slides/S10_FeatureEngineering'
import { S11_ExperimentalSetup } from './slides/S11_ExperimentalSetup'
import { S12_ResultsBaseline } from './slides/S12_ResultsBaseline'
import { S13_ResultsDownsampling } from './slides/S13_ResultsDownsampling'
import { S14_SHAP } from './slides/S14_SHAP'
import { S15_ResultsReduction } from './slides/S15_ResultsReduction'
import { S16_ResultsThreshold } from './slides/S16_ResultsThreshold'
import { S17_ChampionScorecard } from './slides/S17_ChampionScorecard'
import { S18_Synthesis } from './slides/S18_Synthesis'
import { S19_Conclusions } from './slides/S19_Conclusions'
import { S20_LiveDemo } from './slides/S20_LiveDemo'
import { S21_ThankYou } from './slides/S21_ThankYou'

const SLIDES = [
  S01_Title, S02_Roadmap, S03_Challenge, S04_ResearchQuestions,
  S05_Literature, S06_Dataset, S07_Methodology, S08_DataSplit,
  S09_EDA, S10_FeatureEngineering, S11_ExperimentalSetup,
  S12_ResultsBaseline, S13_ResultsDownsampling, S14_SHAP, S15_ResultsReduction, S16_ResultsThreshold,
  S17_ChampionScorecard, S18_Synthesis, S19_Conclusions,
  S20_LiveDemo, S21_ThankYou,
]

function PresentationApp() {
  const { currentSlide, setTotal, totalSlides, next, prev, goTo } = usePresentationStore()
  useKeyNav()
  const { pointer } = usePresenterRemote({ currentSlide, totalSlides, next, prev, goTo })

  useEffect(() => {
    setTotal(SLIDES.length)
  }, [setTotal])

  // Scale slide to fill viewport while preserving 16:9
  const containerRef = useRef<HTMLDivElement>(null)
  const scaleRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const resize = () => {
      if (!containerRef.current || !scaleRef.current) return
      const vw = containerRef.current.clientWidth
      const vh = containerRef.current.clientHeight
      const slideW = 1152
      const slideH = 648
      const scale = Math.min(vw / slideW, vh / slideH)
      scaleRef.current.style.transform = `scale(${scale})`
      scaleRef.current.style.width = `${slideW}px`
      scaleRef.current.style.height = `${slideH}px`
    }
    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [])

  const ActiveSlide = SLIDES[currentSlide]

  return (
    <div
      ref={containerRef}
      className="w-screen h-screen flex items-center justify-center overflow-hidden"
      style={{ background: '#0A1628' }}
    >
      <div
        ref={scaleRef}
        className="relative overflow-hidden"
        style={{ transformOrigin: 'center center', boxShadow: '0 0 80px rgba(8,145,178,0.15)' }}
      >
        <AnimatePresence mode="wait">
          <ActiveSlide key={currentSlide} />
        </AnimatePresence>
        <PointerOverlay pointer={pointer} />
        <NavBar />
      </div>
    </div>
  )
}

export default function App() {
  if (window.location.pathname === '/remote') {
    return <RemoteControlPage />
  }

  return <PresentationApp />
}
