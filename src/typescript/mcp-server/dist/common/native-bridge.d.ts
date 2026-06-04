/**
 * Native bridge to wqm-common Rust functions via napi-rs addon
 *
 * Provides single-source-of-truth implementations shared between
 * the Rust daemon/CLI and the TypeScript MCP server.
 */
export declare function calculateProjectId(projectRoot: string, gitRemote: string | null): string;
export declare function calculateProjectIdWithDisambiguation(projectRoot: string, gitRemote: string | null, disambiguationPath: string | null): string;
export declare function normalizeGitUrl(url: string): string;
export declare function detectGitRemote(projectRoot: string): string | null;
export declare function calculateTenantId(projectRoot: string): string;
export declare function generateIdempotencyKey(itemType: string, op: string, tenantId: string, collection: string, payloadJson: string): string | null;
export declare function computeContentHash(content: string): string;
export declare function tokenize(text: string): string[];
export declare function isValidItemType(s: string): boolean;
export declare function isValidQueueOperation(s: string): boolean;
export declare function isValidQueueStatus(s: string): boolean;
export declare function isValidOperationForType(itemType: string, op: string): boolean;
export declare const COLLECTION_PROJECTS: string;
export declare const COLLECTION_LIBRARIES: string;
export declare const COLLECTION_RULES: string;
export declare const COLLECTION_SCRATCHPAD: string;
export declare const DEFAULT_QDRANT_URL: string;
export declare const DEFAULT_GRPC_PORT: number;
export declare const DEFAULT_BRANCH: string;
export declare const PRIORITY_HIGH: number;
export declare const PRIORITY_NORMAL: number;
export declare const PRIORITY_LOW: number;
export declare const FIELD_TENANT_ID: string;
export declare const FIELD_PROJECT_ID: string;
export declare const FIELD_LIBRARY_NAME: string;
export declare const FIELD_LIBRARY_PATH = "library_path";
export declare const FIELD_BASE_POINT: string;
export declare const FIELD_BRANCH: string;
export declare const FIELD_BRANCHES: string;
export declare const FIELD_FILE_TYPE: string;
export declare const FIELD_FILE_PATH: string;
export declare const FIELD_CONCEPT_TAGS: string;
export declare const FIELD_TAGS: string;
export declare const FIELD_DELETED: string;
export declare const FIELD_CONTENT: string;
export declare const FIELD_TITLE: string;
export declare const FIELD_SOURCE_TYPE: string;
export declare const FIELD_DOCUMENT_ID: string;
export declare const FIELD_ITEM_TYPE: string;
export declare const FIELD_PARENT_UNIT_ID: string;
export declare const ITEM_TYPE_TEXT: string;
export declare const ITEM_TYPE_FILE: string;
export declare const ITEM_TYPE_URL: string;
export declare const ITEM_TYPE_WEBSITE: string;
export declare const ITEM_TYPE_DOC: string;
export declare const ITEM_TYPE_FOLDER: string;
export declare const ITEM_TYPE_TENANT: string;
export declare const ITEM_TYPE_COLLECTION: string;
export declare const ALL_ITEM_TYPES: string[];
export declare const OPERATION_ADD: string;
export declare const OPERATION_UPDATE: string;
export declare const OPERATION_DELETE: string;
export declare const OPERATION_SCAN: string;
export declare const OPERATION_RENAME: string;
export declare const OPERATION_UPLIFT: string;
export declare const OPERATION_RESET: string;
export declare const ALL_OPERATIONS: string[];
//# sourceMappingURL=native-bridge.d.ts.map