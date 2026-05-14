/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#080b14",
          secondary: "#0d1221",
          card: "#111827",
          elevated: "#1a2035",
        },
        accent: {
          blue: "#3b82f6",
          cyan: "#06b6d4",
          green: "#10b981",
          purple: "#8b5cf6",
          amber: "#f59e0b",
          rose: "#f43f5e",
        },
        border: {
          DEFAULT: "#1e2d4a",
          light: "#2d3d5a",
        },
        text: {
          primary: "#e2e8f0",
          secondary: "#94a3b8",
          muted: "#475569",
        }
      },
      fontFamily: {
        sans: ["'DM Sans'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
        display: ["'Syne'", "sans-serif"],
      },
      backgroundImage: {
        "grid-pattern": "linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px)",
        "glow-blue": "radial-gradient(ellipse at center, rgba(59,130,246,0.15) 0%, transparent 70%)",
        "glow-cyan": "radial-gradient(ellipse at center, rgba(6,182,212,0.12) 0%, transparent 70%)",
      },
      backgroundSize: {
        "grid": "40px 40px",
      },
      boxShadow: {
        "card": "0 0 0 1px rgba(30,45,74,0.8), 0 4px 24px rgba(0,0,0,0.4)",
        "glow-blue": "0 0 30px rgba(59,130,246,0.2)",
        "glow-green": "0 0 30px rgba(16,185,129,0.2)",
        "glow-rose": "0 0 30px rgba(244,63,94,0.2)",
      },
      animation: {
        "pulse-slow": "pulse 3s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        }
      }
    },
  },
  plugins: [],
};
