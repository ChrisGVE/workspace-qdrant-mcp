/**
 * MCP tool schema definition for the 'store' tool
 */

export const storeToolDefinition = {
  name: 'store',
  description:
    'Store content or register a project. Use type "scratchpad" for ad-hoc/persistent notes and snippets (the right target for working notes; these are project-scoped and surface automatically in project-scoped search), type "library" (default) to store reference documentation, type "url" to fetch and ingest a web page, or type "project" to register a project directory for file watching and ingestion. Note: omitting type defaults to "library" (which requires libraryName) — pass type:"scratchpad" explicitly for notes.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      type: {
        type: 'string',
        enum: ['library', 'url', 'scratchpad', 'project'],
        description:
          'What to store: "scratchpad" for ad-hoc/persistent notes & snippets (project-scoped), "library" for reference docs (default; requires libraryName), "url" to fetch and ingest a web page, "project" to register a project directory',
      },
      content: {
        type: 'string',
        description: 'Content to store (required for type "library")',
      },
      cwd: {
        type: 'string',
        description:
          'Absolute path of your current working directory. For type "scratchpad", pass this so the note is tagged with the current project (over HTTP the server cannot otherwise detect it) — this is what lets the note surface in project-scoped search. Without it the note falls back to the global tenant.',
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
