import { motion, AnimatePresence } from 'framer-motion'
import type { RemotePointer } from './remoteProtocol'

interface PointerOverlayProps {
  pointer: RemotePointer
}

export function PointerOverlay({ pointer }: PointerOverlayProps) {
  return (
    <AnimatePresence>
      {pointer.visible && (
        <motion.div
          className="absolute z-50 pointer-events-none"
          style={{
            left: `${pointer.x * 100}%`,
            top: `${pointer.y * 100}%`,
            transform: 'translate(-50%, -50%)',
          }}
          initial={{ opacity: 0, scale: 0.6 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.6 }}
        >
          <div
            className="w-7 h-7 rounded-full"
            style={{
              background: 'rgba(239, 68, 68, 0.85)',
              boxShadow: '0 0 0 10px rgba(239, 68, 68, 0.22), 0 0 32px rgba(239, 68, 68, 0.9)',
              border: '2px solid rgba(255,255,255,0.9)',
            }}
          />
        </motion.div>
      )}
    </AnimatePresence>
  )
}
