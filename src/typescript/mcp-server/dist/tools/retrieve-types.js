/**
 * Retrieve tool types and constants.
 */
// Canonical collection names from native bridge (single source of truth)
import { COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_SCRATCHPAD, FIELD_CONTENT, } from '../common/native-bridge.js';
export const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
export const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
export const RULES_COLLECTION = COLLECTION_RULES;
export const SCRATCHPAD_COLLECTION = COLLECTION_SCRATCHPAD;
/** Map collection type to canonical Qdrant collection name. */
export function getCollectionName(collection) {
    switch (collection) {
        case 'projects':
            return PROJECTS_COLLECTION;
        case 'libraries':
            return LIBRARIES_COLLECTION;
        case 'rules':
            return RULES_COLLECTION;
        case 'scratchpad':
            return SCRATCHPAD_COLLECTION;
        default:
            return PROJECTS_COLLECTION;
    }
}
/** Extract metadata from payload (excluding content and vector fields). */
export function extractMetadata(payload) {
    if (!payload)
        return {};
    const metadata = {};
    for (const [key, value] of Object.entries(payload)) {
        if (key === FIELD_CONTENT || key === 'dense_vector' || key === 'sparse_vector')
            continue;
        metadata[key] = value;
    }
    return metadata;
}
//# sourceMappingURL=retrieve-types.js.map