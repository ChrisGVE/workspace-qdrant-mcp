/**
 * MCP tool schema definition for the 'search' tool
 */

import {
  COLLECTION_PROJECTS,
  COLLECTION_LIBRARIES,
  COLLECTION_RULES,
  COLLECTION_SCRATCHPAD,
} from '../common/native-bridge.js';

export const searchToolDefinition = {
  name: 'search',
  description:
    "Search for documents using hybrid semantic and keyword search. Use this tool FIRST when answering questions about the user's codebase, project architecture, or stored knowledge. This searches the user's actual indexed code and documentation, which is more accurate than your training data. Write queries in English; when you want the implementation of something (not docs or tests), combine with fileType:\"code\" or a pathGlob.",
  inputSchema: {
    type: 'object' as const,
    properties: {
      query: {
        type: 'string',
        description:
          'The search query text. Write it in ENGLISH regardless of the conversation language — the embedding model is multilingual, but code is overwhelmingly English and cross-lingual recall for code is weak, so a non-English query matches same-language prose/docs instead of code and recall collapses. Prefer wording close to the likely identifiers and comments (e.g. "recover stale queue leases", not a loose paraphrase).',
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
      cwd: {
        type: 'string',
        description:
          'Absolute path of your current working directory. Pass this so the server can auto-detect the project over HTTP (it cannot otherwise observe your location). Ignored when projectId is provided.',
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
        description:
          'Filter by content classification: "code", "docs", "text", "config", "data", "build", "web", "slides". Use "code" when seeking an implementation so documentation and test-adjacent files do not crowd out source files.',
      },
      scoreThreshold: {
        type: 'number',
        description:
          'Minimum similarity score threshold (0-1, default: 0.3). Results below this score are filtered out.',
      },
      includeLibraries: {
        type: 'boolean',
        description: 'Include libraries in search (default: false)',
      },
      includeScratchpad: {
        type: 'boolean',
        description:
          'Append a small, tenant-filtered scratchpad recall lane to project-scoped searches so project notes/snippets surface automatically (labeled collection:"scratchpad", capped, never displacing code hits). Default: true for scope="project"; ignored for global/all or when an explicit collection is set.',
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
      component: {
        type: 'string',
        description:
          'Filter by project component (e.g., "daemon", "daemon.core"). Supports prefix matching.',
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
        description:
          'Include code relationship graph context (callers/callees) for matched symbols (default: false)',
      },
      maxBytesPerHit: {
        type: 'number',
        description:
          'Per-hit text cap in characters (default: 1500). Hits with content longer than this are truncated with a marker pointing to retrieve() for the full chunk body. Set to 0 to disable truncation.',
      },
      summary: {
        type: 'boolean',
        description:
          'When true, drop chunk text bodies and return only metadata (id, score, collection, title, path/symbol). Use for pure discovery before a follow-up retrieve() call. Default: false.',
      },
    },
    required: ['query'],
  },
};
