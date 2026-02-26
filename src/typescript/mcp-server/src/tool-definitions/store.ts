/**
 * MCP tool schema definition for the 'store' tool
 */

export const storeToolDefinition = {
  name: 'store',
  description:
    'Store content or register a project. Use type "library" (default) to store reference documentation, type "url" to fetch and ingest a web page, type "scratchpad" to save persistent notes/scratch space, or type "project" to register a project directory for file watching and ingestion.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      type: {
        type: 'string',
        enum: ['library', 'url', 'scratchpad', 'project'],
        description:
          'What to store: "library" for reference docs (default), "url" to fetch and ingest a web page, "scratchpad" for persistent notes, "project" to register a project directory',
      },
      content: {
        type: 'string',
        description: 'Content to store (required for type "library")',
      },
      libraryName: {
        type: 'string',
        description: 'Library name (required for type "library" unless forProject is true)',
      },
      forProject: {
        type: 'boolean',
        description:
          'When true, store to libraries collection scoped to the current project. libraryName becomes optional (defaults to "project-refs").',
      },
      path: {
        type: 'string',
        description: 'Project directory path (required for type "project")',
      },
      name: {
        type: 'string',
        description:
          'Project display name (optional for type "project", defaults to directory name)',
      },
      title: {
        type: 'string',
        description: 'Content title (for type "library")',
      },
      url: {
        type: 'string',
        description: 'Source URL (for web content)',
      },
      filePath: {
        type: 'string',
        description: 'Source file path',
      },
      tags: {
        type: 'array',
        items: { type: 'string' },
        description: 'Tags for scratchpad entries',
      },
      sourceType: {
        type: 'string',
        enum: ['user_input', 'web', 'file', 'scratchbook', 'note'],
        description: 'Source type (default: user_input)',
      },
      metadata: {
        type: 'object',
        additionalProperties: { type: 'string' },
        description: 'Additional metadata',
      },
    },
  },
};
