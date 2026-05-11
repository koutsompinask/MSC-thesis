import { useEffect, useRef, useState } from 'react'
import { getRemoteWsUrl, type RemoteMessage, type RemotePointer, type RemoteStatus } from './remoteProtocol'

const send = (socket: WebSocket | null, message: RemoteMessage) => {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message))
  }
}

export function RemoteControlPage() {
  const socketRef = useRef<WebSocket | null>(null)
  const padRef = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<RemoteStatus>('connecting')
  const [currentSlide, setCurrentSlide] = useState(0)
  const [totalSlides, setTotalSlides] = useState(0)

  useEffect(() => {
    const socket = new WebSocket(getRemoteWsUrl())
    socketRef.current = socket

    socket.addEventListener('open', () => {
      setStatus('connected')
      send(socket, { type: 'hello', role: 'remote' })
    })

    socket.addEventListener('close', () => setStatus('disconnected'))
    socket.addEventListener('error', () => setStatus('disconnected'))

    socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data) as RemoteMessage
      if (message.type === 'state') {
        setCurrentSlide(message.currentSlide)
        setTotalSlides(message.totalSlides)
      }
    })

    return () => socket.close()
  }, [])

  const command = (message: RemoteMessage) => send(socketRef.current, message)

  const updatePointer = (event: React.PointerEvent<HTMLDivElement>, visible: boolean) => {
    const rect = padRef.current?.getBoundingClientRect()
    if (!rect) return

    const pointer: RemotePointer = {
      visible,
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
    }

    command({ type: 'pointer', pointer })
  }

  const connected = status === 'connected'

  return (
    <div className="min-h-screen w-screen overflow-hidden bg-[#07111f] text-white flex flex-col">
      <header className="px-5 py-5 border-b border-cyan-900/60">
        <div className="text-xs uppercase tracking-[0.28em] text-cyan-300 font-bold">Thesis Remote</div>
        <div className="mt-2 flex items-end justify-between">
          <div>
            <div className="text-3xl font-black">Slide {totalSlides ? currentSlide + 1 : '-'}</div>
            <div className="text-sm text-slate-400">of {totalSlides || '-'}</div>
          </div>
          <div
            className="px-3 py-1.5 rounded-full text-xs font-bold"
            style={{ background: connected ? '#064E3B' : '#7F1D1D', color: connected ? '#A7F3D0' : '#FECACA' }}
          >
            {status}
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col gap-4 p-5">
        <div className="grid grid-cols-2 gap-3">
          <button
            className="rounded-2xl bg-slate-800 py-6 text-xl font-black active:scale-[0.98] disabled:opacity-40"
            disabled={!connected}
            onClick={() => command({ type: 'command', action: 'prev' })}
          >
            Previous
          </button>
          <button
            className="rounded-2xl bg-cyan-500 py-6 text-xl font-black text-slate-950 active:scale-[0.98] disabled:opacity-40"
            disabled={!connected}
            onClick={() => command({ type: 'command', action: 'next' })}
          >
            Next
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button
            className="rounded-xl bg-slate-900 py-4 font-bold text-slate-300 active:scale-[0.98] disabled:opacity-40"
            disabled={!connected}
            onClick={() => command({ type: 'command', action: 'home' })}
          >
            First Slide
          </button>
          <button
            className="rounded-xl bg-slate-900 py-4 font-bold text-slate-300 active:scale-[0.98] disabled:opacity-40"
            disabled={!connected}
            onClick={() => command({ type: 'command', action: 'end' })}
          >
            Last Slide
          </button>
        </div>

        <section className="flex-1 flex flex-col min-h-0">
          <div className="mb-2 flex items-center justify-between">
            <div className="font-bold text-cyan-200">Laser Pointer</div>
            <div className="text-xs text-slate-500">touch and drag</div>
          </div>
          <div
            ref={padRef}
            className="flex-1 rounded-3xl border border-cyan-800/70 bg-[radial-gradient(circle_at_center,#12385a_0,#07111f_70%)] touch-none relative overflow-hidden"
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId)
              updatePointer(event, true)
            }}
            onPointerMove={(event) => {
              if (event.buttons > 0 || event.pointerType === 'touch') updatePointer(event, true)
            }}
            onPointerUp={(event) => updatePointer(event, false)}
            onPointerCancel={(event) => updatePointer(event, false)}
            onPointerLeave={(event) => updatePointer(event, false)}
          >
            <div className="absolute inset-0 grid place-items-center text-slate-500 text-sm font-bold">
              Pointer Pad
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
