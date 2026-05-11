import { useEffect, useRef, useState } from 'react'
import { fetchExamples } from '../demo/api'
import { type ExampleKey, type FeatureValue, useDemoStore } from '../store/useDemoStore'
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
  const selectedExampleRef = useRef<ExampleKey | null>(null)
  const [status, setStatus] = useState<RemoteStatus>('connecting')
  const [pointer, setPointer] = useState<RemotePointer>(hiddenPointer)
  const { selectedExample, selectExample, runInference, reset } = useDemoStore()

  useEffect(() => {
    currentSlideRef.current = currentSlide
    totalSlidesRef.current = totalSlides
  }, [currentSlide, totalSlides])

  useEffect(() => {
    selectedExampleRef.current = selectedExample
  }, [selectedExample])

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

      if (message.type === 'demo') {
        if (message.action === 'jump') goTo(19)
        if (message.action === 'reset') reset()
        if (message.action === 'select') {
          selectedExampleRef.current = message.example
          selectExample(message.example)
        }
        if (message.action === 'run') {
          const example = selectedExampleRef.current
          if (!example) return

          fetchExamples()
            .then((examples) => runInference(examples[example] as Record<string, FeatureValue>))
            .catch((error) => console.error('[remote] demo run failed', error))
        }
      }

      if (message.type === 'pointer') {
        setPointer(message.pointer)
      }
    })

    return () => socket.close()
  }, [goTo, next, prev, reset, runInference, selectExample, selectedExample])

  useEffect(() => {
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) return

    socket.send(JSON.stringify({ type: 'state', currentSlide, totalSlides } satisfies RemoteMessage))
  }, [currentSlide, totalSlides])

  return { status, pointer }
}
