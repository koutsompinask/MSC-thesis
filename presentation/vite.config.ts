import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The demo API key is attached by the dev server (a Node process), not by the
// browser, so it never ends up in the client bundle. Keep the fallback in sync
// with the default in fastapi/config.py.
const apiKey = process.env.FRAUD_API_KEY ?? 'local-dev-demo-key'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        headers: { 'X-API-Key': apiKey },
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
