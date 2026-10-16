import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// base "./" -> works on GitHub Pages project sites and Vercel alike (relative assets)
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
});
