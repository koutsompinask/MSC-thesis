import { WebSocketServer } from 'ws'

const port = Number(process.env.REMOTE_PORT ?? 4174)
const host = process.env.REMOTE_HOST ?? '0.0.0.0'

const wss = new WebSocketServer({ host, port })

const send = (client, payload) => {
  if (client.readyState === client.OPEN) {
    client.send(JSON.stringify(payload))
  }
}

const broadcast = (payload, except) => {
  for (const client of wss.clients) {
    if (client !== except) send(client, payload)
  }
}

wss.on('connection', (socket, req) => {
  const address = req.socket.remoteAddress ?? 'unknown'
  console.log(`[remote] client connected: ${address}`)

  send(socket, { type: 'server-ready', clients: wss.clients.size })
  broadcast({ type: 'clients', clients: wss.clients.size }, socket)

  socket.on('message', (raw) => {
    try {
      const message = JSON.parse(raw.toString())
      broadcast(message, socket)
    } catch (error) {
      send(socket, { type: 'error', message: 'Invalid JSON message' })
    }
  })

  socket.on('close', () => {
    console.log(`[remote] client disconnected: ${address}`)
    broadcast({ type: 'clients', clients: wss.clients.size })
  })
})

wss.on('listening', () => {
  console.log(`[remote] WebSocket hub listening on ws://${host}:${port}`)
  console.log('[remote] Start Vite with: npm run dev:host')
})

wss.on('error', (error) => {
  console.error(`[remote] failed to start: ${error.message}`)
  process.exitCode = 1
})
