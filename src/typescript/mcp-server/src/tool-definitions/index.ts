/**
 * MCP tool schema definitions for ListTools response.
 * Re-exports all per-tool schemas and assembles the full tool list.
 */

export { searchToolDefinition } from './search.js';
export { retrieveToolDefinition } from './retrieve.js';
export { rulesToolDefinition } from './rules.js';
export { storeToolDefinition } from './store.js';
export { grepToolDefinition } from './grep.js';
export { listToolDefinition } from './list.js';
export { embeddingToolDefinition } from './embedding.js';
export { workspaceIndexToolDefinition } from './workspace-index.js';
export { searchEvalToolDefinition } from './search-eval.js';
export { graphToolDefinition } from './graph.js';

import { searchToolDefinition } from './search.js';
import { retrieveToolDefinition } from './retrieve.js';
import { rulesToolDefinition } from './rules.js';
import { storeToolDefinition } from './store.js';
import { grepToolDefinition } from './grep.js';
import { listToolDefinition } from './list.js';
import { embeddingToolDefinition } from './embedding.js';
import { workspaceIndexToolDefinition } from './workspace-index.js';
import { searchEvalToolDefinition } from './search-eval.js';
import { graphToolDefinition } from './graph.js';

/**
 * Returns the full list of tool definitions for the ListTools MCP response
 */
export function getToolDefinitions() {
  return [
    searchToolDefinition,
    retrieveToolDefinition,
    rulesToolDefinition,
    storeToolDefinition,
    grepToolDefinition,
    listToolDefinition,
    embeddingToolDefinition,
    workspaceIndexToolDefinition,
    searchEvalToolDefinition,
    graphToolDefinition,
  ];
}
