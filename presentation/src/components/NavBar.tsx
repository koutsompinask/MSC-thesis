import { usePresentationStore } from '../store/usePresentationStore'
import { C } from '../design/tokens'

export function NavBar() {
  const { currentSlide, totalSlides, next, prev } = usePresentationStore()
  const progress = totalSlides > 1 ? (currentSlide / (totalSlides - 1)) * 100 : 0

  return (
    <>
      {/* Progress bar */}
      <div
        className="absolute bottom-0 left-0 h-0.5 transition-all duration-300"
        style={{ width: `${progress}%`, background: C.teal, zIndex: 50 }}
      />

      {/* Nav controls */}
      <div
        className="absolute bottom-2 right-4 flex items-center gap-3"
        style={{ zIndex: 50 }}
      >
        <span className="text-xs font-medium" style={{ color: C.textMuted }}>
          {currentSlide + 1} / {totalSlides}
        </span>
        <button
          onClick={prev}
          disabled={currentSlide === 0}
          className="w-7 h-7 rounded flex items-center justify-center text-sm transition-opacity disabled:opacity-30"
          style={{ background: C.navyMid, color: C.white }}
        >
          ‹
        </button>
        <button
          onClick={next}
          disabled={currentSlide === totalSlides - 1}
          className="w-7 h-7 rounded flex items-center justify-center text-sm transition-opacity disabled:opacity-30"
          style={{ background: C.teal, color: C.white }}
        >
          ›
        </button>
      </div>
    </>
  )
}
