/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['IBM Plex Sans', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        ink: {
          50: '#f4f6f8',
          100: '#e8ecef',
          200: '#d0d7de',
          300: '#a8b3bd',
          500: '#5b6b79',
          700: '#2f3a44',
          900: '#12181e',
        },
        signal: {
          ok: '#1f7a4d',
          fail: '#b42318',
          warn: '#b54708',
          live: '#175cd3',
        },
      },
    },
  },
  plugins: [],
}
