import { defineConfig, configDefaults } from "vitest/config";

// tsc emits compiled copies of the test suite into report/lib/compiled/test/
// for the OJS bundle. They're the same suite re-transpiled; exclude them
// so vitest runs each test exactly once.
export default defineConfig({
  test: {
    exclude: [...configDefaults.exclude, "report/lib/compiled/**"],
  },
});
