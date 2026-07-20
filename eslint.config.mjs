import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    ".venv/**",
    "dist/**",
    "out/**",
    "build/**",
    "karaoke copy 2/**",
    "public/pitch-worklet.js",
    "public/soundtouch-processor.js",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
