import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/caffeine-brain/',
  assetsInclude: ['**/*.md']  // Add this line to handle markdown files
})