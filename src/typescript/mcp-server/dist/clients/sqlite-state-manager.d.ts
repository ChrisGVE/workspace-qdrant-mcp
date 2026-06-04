/**
 * SQLite state manager for TypeScript MCP server
 *
 * Thin facade over domain-specific query modules. Provides access to the
 * unified_queue and watch_folders tables with graceful degradation.
 *
 * IMPORTANT: Per ADR-003, the Rust daemon owns the SQLite database and schema.
 * This client reads from SQLite directly and sends all mutations via gRPC
 * to the daemon. It must NOT create tables or run migrations.
 */
import type { UnifiedQueueItem, QueueItemType, QueueOperation, QueueStatus, QueueStats, RegisteredProject, ContentPayload, RulesPayload, LibraryPayload } from '../types/state.js';
import type { DaemonClient } from './daemon-client.js';
export type { UnifiedQueueItem, QueueItemType, QueueOperation, QueueStatus, QueueStats, RegisteredProject, ContentPayload, RulesPayload, LibraryPayload, };
export { generateIdempotencyKey, buildContentPayload, buildRulesPayload, buildLibraryPayload, VALID_ITEM_TYPES, VALID_OPERATIONS, } from './queue-payload-builders.js';
import * as searchEventQueries from './search-event-queries.js';
import * as rulesMirrorQueries from './rules-mirror-queries.js';
import * as scratchpadMirrorQueries from './scratchpad-mirror-queries.js';
import * as trackedFilesQueries from './tracked-files-queries/index.js';
export type { SearchEventInput, SearchEventUpdate } from './search-event-queries.js';
export type { RulesMirrorEntry } from './rules-mirror-queries.js';
export type { TrackedFileEntry, SubmoduleEntry, ComponentEntry, ListTrackedFilesOptions, } from './tracked-files-queries/index.js';
export interface SqliteStateManagerConfig {
    dbPath?: string;
}
export interface EnqueueResult {
    queueId: string;
    isNew: boolean;
    idempotencyKey: string;
}
export interface DegradedQueryResult<T> {
    data: T;
    status: 'ok' | 'degraded';
    reason?: 'database_not_found' | 'table_not_found' | 'database_error' | 'daemon_unavailable' | 'daemon_error';
    message?: string;
}
/**
 * SQLite state manager for MCP server
 *
 * Provides synchronous access to the daemon's SQLite database.
 * Uses better-sqlite3 for fast, synchronous operations.
 */
export declare class SqliteStateManager {
    private db;
    private readonly dbPath;
    private initialized;
    private daemonClient;
    constructor(config?: SqliteStateManagerConfig);
    /** Set the daemon client for gRPC write operations. */
    setDaemonClient(client: DaemonClient | null): void;
    initialize(): DegradedQueryResult<boolean>;
    close(): void;
    isConnected(): boolean;
    getDatabasePath(): string;
    enqueueUnified(itemType: QueueItemType, op: QueueOperation, tenantId: string, collection: string, payload: Record<string, unknown>, priority?: number, branch?: string, metadata?: Record<string, unknown>): Promise<DegradedQueryResult<EnqueueResult | null>>;
    getQueueStats(): DegradedQueryResult<QueueStats | null>;
    getProjectByPath(projectPath: string): DegradedQueryResult<RegisteredProject | null>;
    getProjectById(projectId: string): DegradedQueryResult<RegisteredProject | null>;
    listActiveProjects(): DegradedQueryResult<RegisteredProject[]>;
    logSearchEvent(event: searchEventQueries.SearchEventInput): void;
    updateSearchEvent(eventId: string, update: searchEventQueries.SearchEventUpdate): void;
    getMatchingTags(query: string, collection: string, tenantId?: string): {
        tagId: number;
        tag: string;
        score: number;
    }[];
    getKeywordBasketsForTags(tagIds: number[]): {
        tagId: number;
        keywords: string[];
    }[];
    listTags(collection: string, tenantId?: string, limit?: number): {
        tag: string;
        docCount: number;
        avgScore: number;
    }[];
    getTagHierarchy(collection: string, tenantId?: string): {
        name: string;
        level: number;
        parentName: string | null;
        childCount: number;
    }[];
    getWatchFolderIdByTenantId(tenantId: string): string | null;
    getActiveBasePoints(watchFolderId: string, includeSubmodules?: boolean): string[];
    upsertRulesMirror(entry: rulesMirrorQueries.RulesMirrorEntry): void;
    deleteRulesMirror(ruleId: string): void;
    listRulesMirror(scope?: string, tenantId?: string, limit?: number): rulesMirrorQueries.RulesMirrorEntry[];
    upsertScratchpadMirror(entry: scratchpadMirrorQueries.ScratchpadMirrorEntry): void;
    deleteScratchpadMirror(scratchpadId: string): void;
    listScratchpadMirror(tenantId?: string, limit?: number): scratchpadMirrorQueries.ScratchpadMirrorEntry[];
    listTrackedFiles(options: trackedFilesQueries.ListTrackedFilesOptions): DegradedQueryResult<trackedFilesQueries.TrackedFileEntry[]>;
    countTrackedFiles(options: Omit<trackedFilesQueries.ListTrackedFilesOptions, 'limit'>): number;
    listSubmodules(watchFolderId: string): DegradedQueryResult<trackedFilesQueries.SubmoduleEntry[]>;
    listProjectComponents(watchFolderId: string): DegradedQueryResult<trackedFilesQueries.ComponentEntry[]>;
}
//# sourceMappingURL=sqlite-state-manager.d.ts.map