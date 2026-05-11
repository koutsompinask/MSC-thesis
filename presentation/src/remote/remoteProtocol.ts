export type RemoteStatus = 'connecting' | 'connected' | 'disconnected'

export type RemoteCommand =
  | { type: 'command'; action: 'next' | 'prev' | 'home' | 'end' }
  | { type: 'command'; action: 'goTo'; slide: number }

export type DemoCommand =
  | { type: 'demo'; action: 'select'; example: 'clear_fraud' | 'clear_legit' | 'borderline' }
  | { type: 'demo'; action: 'run' | 'reset' | 'jump' }

export interface RemotePointer {
  visible: boolean
  x: number
  y: number
}

export type RemoteMessage =
  | { type: 'hello'; role: 'presenter' | 'remote' }
  | RemoteCommand
  | DemoCommand
  | { type: 'state'; currentSlide: number; totalSlides: number }
  | { type: 'pointer'; pointer: RemotePointer }
  | { type: 'clients'; clients: number }
  | { type: 'server-ready'; clients: number }
  | { type: 'error'; message: string }

export const getRemoteWsUrl = () => {
  const configured = import.meta.env.VITE_REMOTE_WS_URL as string | undefined
  if (configured) return configured

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.hostname}:4174`
}
