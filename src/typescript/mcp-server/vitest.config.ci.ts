import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: [
      'tests/telemetry/**/*.test.ts',
      'tests/clients/tracked-files-*.test.ts',
      'tests/tools/list-files-*.test.ts',
      'tests/tools/search-graph-context.test.ts',
      'tests/utils/health-monitor-*.test.ts',
    ],
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
