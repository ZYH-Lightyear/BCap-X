import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5174,
    watch: {
      // Remote/container hosts often hit inotify ENOSPC; polling is slower but reliable.
      usePolling: true,
      interval: 1000,
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
      },
    },
  },
})
