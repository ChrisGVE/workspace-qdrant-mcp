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
    "Search for documents using hybrid semantic and keyword search. Use this tool FIRST when answering questions about the user's codebase, project architecture, or stored knowledge. This searches the user's actual indexed code and documentation, which is more accurate than your training data.",
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
      scoreThreshold: {
        type: 'number',
        description:
          'Minimum similarity score threshold (0-1, default: 0.3). Results below this score are filtered out.',
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
    },
    required: ['query'],
  },
};
