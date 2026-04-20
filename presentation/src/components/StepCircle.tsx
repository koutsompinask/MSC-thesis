import { C } from '../design/tokens'

interface StepCircleProps {
  num: number
  label: string
  accent?: string
  dark?: boolean
}

export function StepCircle({ num, label, accent = C.teal, dark = false }: StepCircleProps) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        className="w-9 h-9 rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0"
        style={{ background: accent, color: C.white }}
      >
        {num}
      </div>
      <div
        className="text-xs font-medium text-center leading-tight"
        style={{ color: dark ? C.textMuted : C.textMid, maxWidth: '80px' }}
      >
        {label}
      </div>
    </div>
  )
}
