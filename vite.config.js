import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Forced reload
export default defineConfig({
  plugins: [react()],
})
