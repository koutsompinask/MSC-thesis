/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'navy-dark':   '#0A1628',
        'navy':        '#0D2137',
        'navy-mid':    '#1A3A5C',
        'teal':        '#0891B2',
        'teal-bright': '#06B6D4',
        'teal-pale':   '#DBEAFE',
        'bg-page':     '#EEF4FB',
        'bg-card':     '#FFFFFF',
        'text-dark':   '#1E293B',
        'text-mid':    '#475569',
        'text-muted':  '#94A3B8',
        'amber':       '#D97706',
        'amber-pale':  '#FEF3C7',
        'green-main':  '#059669',
        'green-pale':  '#D1FAE5',
        'red-main':    '#DC2626',
        'red-pale':    '#FEE2E2',
        'purple':      '#7C3AED',
        'purple-pale': '#EDE9FE',
      },
      fontFamily: {
        sans: ['Inter', 'Calibri', 'Trebuchet MS', 'system-ui', 'sans-serif'],
      },
      aspectRatio: {
        '16/9': '16 / 9',
      },
    },
  },
  plugins: [],
}
