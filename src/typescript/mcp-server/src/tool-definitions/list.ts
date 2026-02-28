/**
 * MCP tool schema definition for the 'list' tool
 */

export const listToolDefinition = {
  name: 'list',
  description:
    'List project files and folder structure. Shows only indexed files (excludes gitignored, node_modules, etc). Use format "summary" first to understand project layout, then drill into specific folders with the path parameter.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      path: {
        type: 'string',
        description: 'Subfolder relative to project root (default: root)',
      },
      depth: {
        type: 'number',
        description: 'Max directory depth (default: 3, max: 10)',
      },
      format: {
        type: 'string',
        enum: ['tree', 'summary', 'flat'],
        description: 'Output format (default: tree)',
      },
      fileType: {
        type: 'string',
        description: 'Filter: "code", "text", "data", "config", "build", "web"',
      },
      language: {
        type: 'string',
        description: 'Filter by programming language (e.g., "rust", "typescript")',
      },
      extension: {
        type: 'string',
        description: 'Filter by file extension (e.g., "rs", "ts")',
      },
      pattern: {
        type: 'string',
        description: 'Glob pattern on relative path (e.g., "**/*.test.ts")',
      },
      includeTests: {
        type: 'boolean',
        description: 'Include test files (default: true)',
      },
      limit: {
        type: 'number',
        description: 'Max entries returned (default: 200, max: 500)',
      },
      projectId: {
        type: 'string',
        description: 'Specific project ID (default: current project)',
      },
      component: {
        type: 'string',
        description:
          'Filter by component (dot-separated ID or prefix, e.g. "daemon" or "daemon.core"). Auto-detected from Cargo.toml/package.json workspaces.',
      },
    },
  },
};
