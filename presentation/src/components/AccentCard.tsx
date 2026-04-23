import type { ReactNode } from 'react'
import { C } from '../design/tokens'

interface AccentCardProps {
  accent?: string
  children: ReactNode
  className?: string
  title?: string
}

export function AccentCard({ accent = C.teal, children, className = '', title }: AccentCardProps) {
  return (
    <div
      className={`flex rounded-lg overflow-hidden ${className}`}
      style={{
        background: C.bgCard,
        boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
        border: '1px solid #e2e8f0',
      }}
    >
      <div className="w-1.5 flex-shrink-0" style={{ background: accent }} />
      <div className="flex-1 p-4">
        {title && (
          <div className="font-semibold text-base mb-1.5 leading-snug" style={{ color: C.textDark }}>
            {title}
          </div>
        )}
        {children}
      </div>
    </div>
  )
}
