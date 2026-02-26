/**
 * MCP tool schema definition for the 'retrieve' tool
 */

import {
  COLLECTION_PROJECTS,
  COLLECTION_LIBRARIES,
  COLLECTION_RULES,
  COLLECTION_SCRATCHPAD,
} from '../common/native-bridge.js';

export const retrieveToolDefinition = {
  name: 'retrieve',
  description:
    'Retrieve documents by ID or metadata filter. Use this to access specific documents when you know the document ID. Prefer `search` for discovery, `retrieve` for known documents.',
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
};
