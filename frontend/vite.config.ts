import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev: the API is proxied to the FastAPI server (uvicorn backend.app:app --port 8000).
// Build: `frontend/dist` is served by the same FastAPI process.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
