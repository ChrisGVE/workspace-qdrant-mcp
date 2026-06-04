/**
 * Search tool types, interfaces, and constants.
 */
// Canonical collection names from native bridge (single source of truth)
import { COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_SCRATCHPAD, COLLECTION_RULES, } from '../common/native-bridge.js';
export const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
export const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
export const SCRATCHPAD_COLLECTION = COLLECTION_SCRATCHPAD;
export const RULES_COLLECTION = COLLECTION_RULES;
// Vector names for hybrid search
export const DENSE_VECTOR_NAME = 'dense';
export const SPARSE_VECTOR_NAME = 'sparse';
// RRF constant (k=60 is standard)
export const RRF_K = 60;
// Default search parameters
export const DEFAULT_LIMIT = 10;
export const DEFAULT_SCORE_THRESHOLD = 0.3;
// Tag expansion defaults
export const DEFAULT_EXPANSION_WEIGHT = 0.5;
export const DEFAULT_MAX_EXPANDED_KEYWORDS = 10;
//# sourceMappingURL=search-types.js.map