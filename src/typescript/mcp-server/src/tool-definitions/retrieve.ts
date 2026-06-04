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
    "Retrieve documents by their point id or by a metadata filter. Pass `documentId` = the `id` field from a search/list result (NOT the metadata `document_id`). To look up by the `document_id` metadata field instead, use `filter: {\"document_id\": \"...\"}`. Prefer `search` for discovery, `retrieve` for known points.",
  inputSchema: {
    type: 'object' as const,
    properties: {
      documentId: {
        type: 'string',
        description:
          'The point id to retrieve — the `id` field from a search or list result (a Qdrant point UUID). NOT the metadata `document_id` (a content hash); to match that, use `filter: {"document_id": "..."}` instead.',
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
      cwd: {
        type: 'string',
        description:
          'Absolute path of your current working directory. Pass this so the server can auto-detect the project over HTTP (it cannot otherwise observe your location). Ignored when projectId is provided.',
      },
      libraryName: {
        type: 'string',
        description: 'Library name for libraries collection',
      },
    },
  },
};
