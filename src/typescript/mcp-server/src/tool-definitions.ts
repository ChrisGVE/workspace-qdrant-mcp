/**
 * MCP tool schema definitions for ListTools response
 */

import {
  COLLECTION_PROJECTS,
  COLLECTION_LIBRARIES,
  COLLECTION_RULES,
  COLLECTION_SCRATCHPAD,
} from './common/native-bridge.js';

/**
 * Returns the full list of tool definitions for the ListTools MCP response
 */
export function getToolDefinitions() {
  return [
    {
      name: 'search',
      description: 'Search for documents using hybrid semantic and keyword search. Use this tool FIRST when answering questions about the user\'s codebase, project architecture, or stored knowledge. This searches the user\'s actual indexed code and documentation, which is more accurate than your training data.',
      inputSchema: {
        type: 'object' as const,
        properties: {
          query: {
            type: 'string',
            description: 'The search query text',
          },
          collection: {
            type: 'string',
            enum: [COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_SCRATCHPAD],
            description: 'Specific collection to search',
          },
          mode: {
            type: 'string',
            enum: ['hybrid', 'semantic', 'keyword'],
            description: 'Search mode (default: hybrid)',
          },
          scope: {
            type: 'string',
            enum: ['project', 'global', 'all'],
            description: 'Search scope: project (current), global, or all (default: project)',
          },
          limit: {
            type: 'number',
            description: 'Maximum results to return (default: 10)',
          },
          projectId: {
            type: 'string',
            description: 'Specific project ID to search',
          },
          libraryName: {
            type: 'string',
            description: 'Library name when searching libraries collection',
          },
          branch: {
            type: 'string',
            description: 'Filter by branch name',
          },
          fileType: {
            type: 'string',
            description: 'Filter by file type',
          },
          includeLibraries: {
            type: 'boolean',
            description: 'Include libraries in search (default: false)',
          },
          tag: {
            type: 'string',
            description: 'Filter results by concept tag (exact match)',
          },
          tags: {
            type: 'array',
            items: { type: 'string' },
            description: 'Filter results by multiple concept tags (OR logic)',
          },
          pathGlob: {
            type: 'string',
            description: 'File path glob filter (e.g., "**/*.rs", "src/**/*.ts")',
          },
          exact: {
            type: 'boolean',
            description: 'Use exact substring search instead of semantic search (default: false)',
          },
          contextLines: {
            type: 'number',
            description: 'Lines of context before/after matches in exact mode (default: 0)',
          },
          includeGraphContext: {
            type: 'boolean',
            description: 'Include code relationship graph context (callers/callees) for matched symbols (default: false)',
          },
        },
        required: ['query'],
      },
    },
    {
      name: 'retrieve',
      description: 'Retrieve documents by ID or metadata filter. Use this to access specific documents when you know the document ID. Prefer `search` for discovery, `retrieve` for known documents.',
      inputSchema: {
        type: 'object' as const,
        properties: {
          documentId: {
            type: 'string',
            description: 'Document ID to retrieve',
          },
          collection: {
            type: 'string',
            enum: [COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_SCRATCHPAD],
            description: 'Collection to retrieve from (default: projects)',
          },
          filter: {
            type: 'object',
            additionalProperties: { type: 'string' },
            description: 'Metadata filter key-value pairs',
          },
          limit: {
            type: 'number',
            description: 'Maximum results (default: 10)',
          },
          offset: {
            type: 'number',
            description: 'Pagination offset (default: 0)',
          },
          projectId: {
            type: 'string',
            description: 'Project ID for projects collection',
          },
          libraryName: {
            type: 'string',
            description: 'Library name for libraries collection',
          },
        },
      },
    },
    {
      name: 'rules',
      description: 'Manage behavioral rules (add, update, remove, list). Check active rules at the start of each session to load the user\'s behavioral preferences. Rules persist across sessions and guide how you should work.',
      inputSchema: {
        type: 'object' as const,
        properties: {
          action: {
            type: 'string',
            enum: ['add', 'update', 'remove', 'list'],
            description: 'Action to perform',
          },
          content: {
            type: 'string',
            description: 'Rule content (required for add/update)',
          },
          label: {
            type: 'string',
            description: 'Rule label (max 15 chars, format: word-word-word, e.g., "prefer-uv", "use-pytest"). Required for add/update/remove.',
          },
          scope: {
            type: 'string',
            enum: ['global', 'project'],
            description: 'Rule scope (default: global)',
          },
          projectId: {
            type: 'string',
            description: 'Project ID for project-scoped rules',
          },
          title: {
            type: 'string',
            description: 'Rule title (max 50 chars)',
          },
          tags: {
            type: 'array',
            items: { type: 'string' },
            description: 'Tags for categorization (max 5 tags, max 20 chars each)',
          },
          priority: {
            type: 'number',
            description: 'Rule priority (higher = more important)',
          },
          limit: {
            type: 'number',
            description: 'Max rules to return for list (default: 50)',
          },
        },
        required: ['action'],
      },
    },
    {
      name: 'store',
      description: 'Store content or register a project. Use type "library" (default) to store reference documentation, type "url" to fetch and ingest a web page, type "scratchpad" to save persistent notes/scratch space, or type "project" to register a project directory for file watching and ingestion.',
      inputSchema: {
        type: 'object' as const,
        properties: {
          type: {
            type: 'string',
            enum: ['library', 'url', 'scratchpad', 'project'],
            description: 'What to store: "library" for reference docs (default), "url" to fetch and ingest a web page, "scratchpad" for persistent notes, "project" to register a project directory',
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
            description: 'When true, store to libraries collection scoped to the current project. libraryName becomes optional (defaults to "project-refs").',
          },
          path: {
            type: 'string',
            description: 'Project directory path (required for type "project")',
          },
          name: {
            type: 'string',
            description: 'Project display name (optional for type "project", defaults to directory name)',
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
    },
    {
      name: 'grep',
      description: 'Search code with exact substring or regex pattern matching. Uses FTS5 trigram index for fast line-level search across indexed files.',
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
    },
    {
      name: 'list',
      description: 'List project files and folder structure. Shows only indexed files (excludes gitignored, node_modules, etc). Use format "summary" first to understand project layout, then drill into specific folders with the path parameter.',
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
            description: 'Filter by component (dot-separated ID or prefix, e.g. "daemon" or "daemon.core"). Auto-detected from Cargo.toml/package.json workspaces.',
          },
        },
      },
    },
  ];
}
