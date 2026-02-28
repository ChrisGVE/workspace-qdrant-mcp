/**
 * MCP tool schema definition for the 'grep' tool
 */

export const grepToolDefinition = {
  name: 'grep',
  description:
    'Search code with exact substring or regex pattern matching. Uses FTS5 trigram index for fast line-level search across indexed files.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      pattern: {
        type: 'string',
        description: 'Search pattern (exact substring or regex)',
      },
      regex: {
        type: 'boolean',
        description: 'Treat pattern as regex (default: false)',
      },
      caseSensitive: {
        type: 'boolean',
        description: 'Case-sensitive matching (default: true)',
      },
      pathGlob: {
        type: 'string',
        description: 'File path glob filter (e.g., "**/*.rs", "src/**/*.ts")',
      },
      scope: {
        type: 'string',
        enum: ['project', 'all'],
        description: 'Search scope: project (current) or all (default: project)',
      },
      contextLines: {
        type: 'number',
        description: 'Lines of context before/after each match (default: 0)',
      },
      maxResults: {
        type: 'number',
        description: 'Maximum results to return (default: 1000)',
      },
      branch: {
        type: 'string',
        description: 'Filter by branch name',
      },
      projectId: {
        type: 'string',
        description: 'Specific project ID to search',
      },
    },
    required: ['pattern'],
  },
};
