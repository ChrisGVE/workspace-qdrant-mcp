import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    exclude: [
      'tests/integration/**',
      'tests/server-*.test.ts',
      'tests/clients/sqlite-state-manager-*.test.ts',
      'tests/utils/project-detector-*.test.ts',
      'tests/tools/retrieve-*.test.ts',
      'tests/tools/rules-*.test.ts',
      'tests/tools/search-base-points-*.test.ts',
      'tests/tools/search-exact-*.test.ts',
      'tests/tools/search-expansion-*.test.ts',
      'tests/tools/search-filters*.test.ts',
      'tests/tools/search-modes-*.test.ts',
      'tests/tools/search-result-*.test.ts',
    ],
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
