/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          lightBlue: '#E6F2FC',       // Main background
          cyan: '#00A9D6',            // Primary buttons & active elements
          cyanHover: '#008FB5',       // Button hover state
          navy: '#203B5D',            // Headings & primary text
          navyMuted: '#38577D',       // Supporting navy
          white: '#FFFFFF',           // Cards & surfaces
          softBlue: '#D5E8F8',        // Secondary background
          softBlueBorder: '#C5DCF2',  // Subtle borders
          slate: '#71849A',           // Descriptions & secondary text
          slateLight: '#94A3B8',      // Muted labels

          // Sentiment System (Clean Academic Palette)
          positive: '#00A9D6',        // Cyan highlight
          neutral: '#71849A',         // Slate grey
          negative: '#203B5D',        // Deep navy
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
