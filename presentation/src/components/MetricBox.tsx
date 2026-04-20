import { C } from '../design/tokens'

interface MetricBoxProps {
  value: string
  label: string
  accent?: string
  dark?: boolean
}

export function MetricBox({ value, label, accent = C.teal, dark = false }: MetricBoxProps) {
  return (
    <div
      className="flex flex-col items-center justify-center rounded-lg px-4 py-3 gap-0.5"
      style={{
        background: dark ? C.navyMid : C.bgCard,
        border: `1px solid ${accent}33`,
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
      }}
    >
      <div
        className="text-2xl font-bold leading-none"
        style={{ color: accent }}
      >
        {value}
      </div>
      <div
        className="text-xs font-medium uppercase tracking-wide text-center"
        style={{ color: dark ? C.textMuted : C.textMid }}
      >
        {label}
      </div>
    </div>
  )
}
