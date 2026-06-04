/**
 * Native bridge to wqm-common Rust functions via napi-rs addon
 *
 * Provides single-source-of-truth implementations shared between
 * the Rust daemon/CLI and the TypeScript MCP server.
 */
import { createRequire } from 'node:module';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
// The native addon is located relative to this file:
// src/typescript/mcp-server/src/common/native-bridge.ts
// -> src/rust/common-node/index.js
const NATIVE_ADDON_PATH = join(__dirname, '..', '..', '..', '..', 'rust', 'common-node', 'index.js');
let addon = null;
function loadAddon() {
    if (addon)
        return addon;
    try {
        addon = require(NATIVE_ADDON_PATH);
        return addon;
    }
    catch (err) {
        throw new Error(`Failed to load wqm-common-node native addon from ${NATIVE_ADDON_PATH}. ` +
            `Build it with: cd src/rust/common-node && napi build --release --platform. ` +
            `Original error: ${err instanceof Error ? err.message : String(err)}`);
    }
}
// Re-export functions
export function calculateProjectId(projectRoot, gitRemote) {
    return loadAddon().calculateProjectId(projectRoot, gitRemote);
}
export function calculateProjectIdWithDisambiguation(projectRoot, gitRemote, disambiguationPath) {
    return loadAddon().calculateProjectIdWithDisambiguation(projectRoot, gitRemote, disambiguationPath);
}
export function normalizeGitUrl(url) {
    return loadAddon().normalizeGitUrl(url);
}
export function detectGitRemote(projectRoot) {
    return loadAddon().detectGitRemote(projectRoot);
}
export function calculateTenantId(projectRoot) {
    return loadAddon().calculateTenantId(projectRoot);
}
export function generateIdempotencyKey(itemType, op, tenantId, collection, payloadJson) {
    return loadAddon().generateIdempotencyKey(itemType, op, tenantId, collection, payloadJson);
}
export function computeContentHash(content) {
    return loadAddon().computeContentHash(content);
}
export function tokenize(text) {
    return loadAddon().tokenize(text);
}
export function isValidItemType(s) {
    return loadAddon().isValidItemType(s);
}
export function isValidQueueOperation(s) {
    return loadAddon().isValidQueueOperation(s);
}
export function isValidQueueStatus(s) {
    return loadAddon().isValidQueueStatus(s);
}
export function isValidOperationForType(itemType, op) {
    return loadAddon().isValidOperationForType(itemType, op);
}
// Eagerly resolve constants (called once on import)
export const COLLECTION_PROJECTS = loadAddon().collectionProjects();
export const COLLECTION_LIBRARIES = loadAddon().collectionLibraries();
export const COLLECTION_RULES = loadAddon().collectionRules();
export const COLLECTION_SCRATCHPAD = loadAddon().collectionScratchpad();
export const DEFAULT_QDRANT_URL = loadAddon().defaultQdrantUrl();
export const DEFAULT_GRPC_PORT = loadAddon().defaultGrpcPort();
export const DEFAULT_BRANCH = loadAddon().defaultBranch();
export const PRIORITY_HIGH = loadAddon().priorityHigh();
export const PRIORITY_NORMAL = loadAddon().priorityNormal();
export const PRIORITY_LOW = loadAddon().priorityLow();
// Payload field constants
export const FIELD_TENANT_ID = loadAddon().fieldTenantId();
export const FIELD_PROJECT_ID = loadAddon().fieldProjectId();
export const FIELD_LIBRARY_NAME = loadAddon().fieldLibraryName();
export const FIELD_LIBRARY_PATH = 'library_path';
export const FIELD_BASE_POINT = loadAddon().fieldBasePoint();
export const FIELD_BRANCH = loadAddon().fieldBranch();
export const FIELD_BRANCHES = loadAddon().fieldBranches();
export const FIELD_FILE_TYPE = loadAddon().fieldFileType();
export const FIELD_FILE_PATH = loadAddon().fieldFilePath();
export const FIELD_CONCEPT_TAGS = loadAddon().fieldConceptTags();
export const FIELD_TAGS = loadAddon().fieldTags();
export const FIELD_DELETED = loadAddon().fieldDeleted();
export const FIELD_CONTENT = loadAddon().fieldContent();
export const FIELD_TITLE = loadAddon().fieldTitle();
export const FIELD_SOURCE_TYPE = loadAddon().fieldSourceType();
export const FIELD_DOCUMENT_ID = loadAddon().fieldDocumentId();
export const FIELD_ITEM_TYPE = loadAddon().fieldItemType();
export const FIELD_PARENT_UNIT_ID = loadAddon().fieldParentUnitId();
// Item type constants
export const ITEM_TYPE_TEXT = loadAddon().itemTypeText();
export const ITEM_TYPE_FILE = loadAddon().itemTypeFile();
export const ITEM_TYPE_URL = loadAddon().itemTypeUrl();
export const ITEM_TYPE_WEBSITE = loadAddon().itemTypeWebsite();
export const ITEM_TYPE_DOC = loadAddon().itemTypeDoc();
export const ITEM_TYPE_FOLDER = loadAddon().itemTypeFolder();
export const ITEM_TYPE_TENANT = loadAddon().itemTypeTenant();
export const ITEM_TYPE_COLLECTION = loadAddon().itemTypeCollection();
export const ALL_ITEM_TYPES = loadAddon().allItemTypes();
// Operation constants
export const OPERATION_ADD = loadAddon().operationAdd();
export const OPERATION_UPDATE = loadAddon().operationUpdate();
export const OPERATION_DELETE = loadAddon().operationDelete();
export const OPERATION_SCAN = loadAddon().operationScan();
export const OPERATION_RENAME = loadAddon().operationRename();
export const OPERATION_UPLIFT = loadAddon().operationUplift();
export const OPERATION_RESET = loadAddon().operationReset();
export const ALL_OPERATIONS = loadAddon().allOperations();
//# sourceMappingURL=native-bridge.js.map