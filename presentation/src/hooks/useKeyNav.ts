import { useEffect } from 'react'
import { usePresentationStore } from '../store/usePresentationStore'

export function useKeyNav() {
  const { next, prev, goTo, totalSlides } = usePresentationStore()

  useEffect(() => {
    let gBuffer = ''
    let gTimer: ReturnType<typeof setTimeout> | null = null

    const handle = (e: KeyboardEvent) => {
      // G + number to jump: press G, then type a number
      if (e.key === 'g' || e.key === 'G') {
        gBuffer = ''
        return
      }
      if (gBuffer !== undefined && /^[0-9]$/.test(e.key)) {
        gBuffer += e.key
        if (gTimer) clearTimeout(gTimer)
        gTimer = setTimeout(() => {
          if (gBuffer) goTo(parseInt(gBuffer) - 1)
          gBuffer = ''
        }, 600)
        return
      }

      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowDown':
        case ' ':
          e.preventDefault()
          next()
          break
        case 'ArrowLeft':
        case 'ArrowUp':
          e.preventDefault()
          prev()
          break
        case 'Home':
          goTo(0)
          break
        case 'End':
          goTo(totalSlides - 1)
          break
      }
    }

    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [next, prev, goTo, totalSlides])
}
