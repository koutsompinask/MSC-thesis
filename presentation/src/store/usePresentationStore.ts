import { create } from 'zustand'

interface PresentationStore {
  currentSlide: number
  totalSlides: number
  setTotal: (n: number) => void
  next: () => void
  prev: () => void
  goTo: (n: number) => void
}

export const usePresentationStore = create<PresentationStore>()((set) => ({
  currentSlide: 0,
  totalSlides: 0,
  setTotal: (n: number) => set({ totalSlides: n }),
  next: () => set((s: PresentationStore) => ({ currentSlide: Math.min(s.currentSlide + 1, s.totalSlides - 1) })),
  prev: () => set((s: PresentationStore) => ({ currentSlide: Math.max(s.currentSlide - 1, 0) })),
  goTo: (n: number) => set((s: PresentationStore) => ({ currentSlide: Math.max(0, Math.min(n, s.totalSlides - 1)) })),
}))
