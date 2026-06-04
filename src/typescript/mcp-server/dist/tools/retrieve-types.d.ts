/**
 * Retrieve tool types and constants.
 */
export declare const PROJECTS_COLLECTION: string;
export declare const LIBRARIES_COLLECTION: string;
export declare const RULES_COLLECTION: string;
export declare const SCRATCHPAD_COLLECTION: string;
export type RetrieveCollectionType = 'projects' | 'libraries' | 'rules' | 'scratchpad';
export interface RetrieveOptions {
    documentId?: string;
    collection?: RetrieveCollectionType;
    filter?: Record<string, string>;
    limit?: number;
    offset?: number;
    projectId?: string;
    libraryName?: string;
}
export interface RetrievedDocument {
    id: string;
    content: string;
    metadata: Record<string, unknown>;
    score?: number;
}
export interface RetrieveResponse {
    success: boolean;
    documents: RetrievedDocument[];
    total?: number;
    hasMore?: boolean;
    message?: string;
}
export interface RetrieveToolConfig {
    qdrantUrl: string;
    qdrantApiKey?: string;
    qdrantTimeout?: number;
}
/** Map collection type to canonical Qdrant collection name. */
export declare function getCollectionName(collection: RetrieveCollectionType): string;
/** Extract metadata from payload (excluding content and vector fields). */
export declare function extractMetadata(payload: Record<string, unknown> | null | undefined): Record<string, unknown>;
//# sourceMappingURL=retrieve-types.d.ts.map