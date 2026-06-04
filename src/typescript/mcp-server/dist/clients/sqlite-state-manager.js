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
import Database, {} from 'better-sqlite3';
import { existsSync } from 'node:fs';
import { getDatabasePath } from '../utils/paths.js';
import { PRIORITY_LOW } from '../common/native-bridge.js';
// Re-export payload builder functions
export { generateIdempotencyKey, buildContentPayload, buildRulesPayload, buildLibraryPayload, VALID_ITEM_TYPES, VALID_OPERATIONS, } from './queue-payload-builders.js';
// Import delegate modules
import * as queueOps from './queue-operations.js';
import * as projectQueries from './project-queries.js';
import * as searchEventQueries from './search-event-queries.js';
import * as tagQueries from './tag-queries.js';
import * as instanceQueries from './instance-queries.js';
import * as rulesMirrorQueries from './rules-mirror-queries.js';
import * as scratchpadMirrorQueries from './scratchpad-mirror-queries.js';
import * as trackedFilesQueries from './tracked-files-queries/index.js';
const DEFAULT_DB_PATH = getDatabasePath();
/**
 * SQLite state manager for MCP server
 *
 * Provides synchronous access to the daemon's SQLite database.
 * Uses better-sqlite3 for fast, synchronous operations.
 */
export class SqliteStateManager {
    db = null;
    dbPath;
    initialized = false;
    daemonClient = null;
    constructor(config = {}) {
        this.dbPath = config.dbPath ?? DEFAULT_DB_PATH;
    }
    /** Set the daemon client for gRPC write operations. */
    setDaemonClient(client) {
        this.daemonClient = client;
    }
    // ── Core lifecycle ────────────────────────────────────────────────────
    initialize() {
        if (this.initialized)
            return { data: true, status: 'ok' };
        if (!existsSync(this.dbPath)) {
            return {
                data: false,
                status: 'degraded',
                reason: 'database_not_found',
                message: `Database not found at ${this.dbPath}. Daemon has not initialized yet.`,
            };
        }
        try {
            this.db = new Database(this.dbPath, { readonly: true, fileMustExist: true });
            this.initialized = true;
            return { data: true, status: 'ok' };
        }
        catch (error) {
            return {
                data: false,
                status: 'degraded',
                reason: 'database_error',
                message: `Failed to open database: ${error instanceof Error ? error.message : 'Unknown error'}`,
            };
        }
    }
    close() {
        if (this.db) {
            this.db.close();
            this.db = null;
            this.initialized = false;
        }
    }
    isConnected() {
        return this.initialized && this.db !== null;
    }
    getDatabasePath() {
        return this.dbPath;
    }
    // ── Unified Queue (delegated) ─────────────────────────────────────────
    enqueueUnified(itemType, op, tenantId, collection, payload, priority = PRIORITY_LOW, branch = 'main', metadata) {
        return queueOps.enqueueUnified(this.daemonClient, itemType, op, tenantId, collection, payload, priority, branch, metadata);
    }
    getQueueStats() {
        return queueOps.getQueueStats(this.db);
    }
    // ── Project queries (delegated) ───────────────────────────────────────
    getProjectByPath(projectPath) {
        return projectQueries.getProjectByPath(this.db, projectPath);
    }
    getProjectById(projectId) {
        return projectQueries.getProjectById(this.db, projectId);
    }
    listActiveProjects() {
        return projectQueries.listActiveProjects(this.db);
    }
    // ── Search event instrumentation (delegated) ──────────────────────────
    logSearchEvent(event) {
        searchEventQueries.logSearchEvent(this.daemonClient, event);
    }
    updateSearchEvent(eventId, update) {
        searchEventQueries.updateSearchEvent(this.daemonClient, eventId, update);
    }
    // ── Tag/basket queries (delegated) ────────────────────────────────────
    getMatchingTags(query, collection, tenantId) {
        return tagQueries.getMatchingTags(this.db, query, collection, tenantId);
    }
    getKeywordBasketsForTags(tagIds) {
        return tagQueries.getKeywordBasketsForTags(this.db, tagIds);
    }
    listTags(collection, tenantId, limit = 50) {
        return tagQueries.listTags(this.db, collection, tenantId, limit);
    }
    getTagHierarchy(collection, tenantId) {
        return tagQueries.getTagHierarchy(this.db, collection, tenantId);
    }
    // ── Instance-aware queries (delegated) ────────────────────────────────
    getWatchFolderIdByTenantId(tenantId) {
        return instanceQueries.getWatchFolderIdByTenantId(this.db, tenantId);
    }
    getActiveBasePoints(watchFolderId, includeSubmodules = false) {
        return instanceQueries.getActiveBasePoints(this.db, watchFolderId, includeSubmodules);
    }
    // ── Rules mirror (delegated) ──────────────────────────────────────────
    upsertRulesMirror(entry) {
        rulesMirrorQueries.upsertRulesMirror(this.daemonClient, entry);
    }
    deleteRulesMirror(ruleId) {
        rulesMirrorQueries.deleteRulesMirror(this.daemonClient, ruleId);
    }
    listRulesMirror(scope, tenantId, limit = 50) {
        return rulesMirrorQueries.listRulesMirror(this.db, scope, tenantId, limit);
    }
    // ── Scratchpad mirror (delegated) ──────────────────────────────────────
    upsertScratchpadMirror(entry) {
        scratchpadMirrorQueries.upsertScratchpadMirror(this.daemonClient, entry);
    }
    deleteScratchpadMirror(scratchpadId) {
        scratchpadMirrorQueries.deleteScratchpadMirror(this.daemonClient, scratchpadId);
    }
    listScratchpadMirror(tenantId, limit = 100) {
        return scratchpadMirrorQueries.listScratchpadMirror(this.db, tenantId, limit);
    }
    // ── Tracked files (delegated) ──────────────────────────────────────────
    listTrackedFiles(options) {
        return trackedFilesQueries.listTrackedFiles(this.db, options);
    }
    countTrackedFiles(options) {
        return trackedFilesQueries.countTrackedFiles(this.db, options);
    }
    listSubmodules(watchFolderId) {
        return trackedFilesQueries.listSubmodules(this.db, watchFolderId);
    }
    listProjectComponents(watchFolderId) {
        return trackedFilesQueries.listProjectComponents(this.db, watchFolderId);
    }
}
//# sourceMappingURL=sqlite-state-manager.js.map