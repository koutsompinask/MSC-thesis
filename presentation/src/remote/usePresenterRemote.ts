import { useEffect, useRef, useState } from 'react'
import { getRemoteWsUrl, type RemoteMessage, type RemotePointer, type RemoteStatus } from './remoteProtocol'

interface UsePresenterRemoteOptions {
  currentSlide: number
  totalSlides: number
  next: () => void
  prev: () => void
  goTo: (slide: number) => void
}

const hiddenPointer: RemotePointer = { visible: false, x: 0.5, y: 0.5 }

export function usePresenterRemote({ currentSlide, totalSlides, next, prev, goTo }: UsePresenterRemoteOptions) {
  const socketRef = useRef<WebSocket | null>(null)
  const currentSlideRef = useRef(currentSlide)
  const totalSlidesRef = useRef(totalSlides)
  const [status, setStatus] = useState<RemoteStatus>('connecting')
  const [pointer, setPointer] = useState<RemotePointer>(hiddenPointer)

  useEffect(() => {
    currentSlideRef.current = currentSlide
    totalSlidesRef.current = totalSlides
  }, [currentSlide, totalSlides])

  useEffect(() => {
    const socket = new WebSocket(getRemoteWsUrl())
    socketRef.current = socket

    socket.addEventListener('open', () => {
      setStatus('connected')
      socket.send(JSON.stringify({ type: 'hello', role: 'presenter' } satisfies RemoteMessage))
    })

    socket.addEventListener('close', () => setStatus('disconnected'))
    socket.addEventListener('error', () => setStatus('disconnected'))

    socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data) as RemoteMessage

      if (message.type === 'hello' && message.role === 'remote') {
        socket.send(JSON.stringify({
          type: 'state',
          currentSlide: currentSlideRef.current,
          totalSlides: totalSlidesRef.current,
        } satisfies RemoteMessage))
      }

      if (message.type === 'command') {
        if (message.action === 'next') next()
        if (message.action === 'prev') prev()
        if (message.action === 'home') goTo(0)
        if (message.action === 'end') goTo(totalSlidesRef.current - 1)
        if (message.action === 'goTo') goTo(message.slide)
      }

      if (message.type === 'pointer') {
        setPointer(message.pointer)
      }
    })

    return () => socket.close()
  }, [goTo, next, prev])

  useEffect(() => {
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) return

    socket.send(JSON.stringify({ type: 'state', currentSlide, totalSlides } satisfies RemoteMessage))
  }, [currentSlide, totalSlides])

  return { status, pointer }
}
