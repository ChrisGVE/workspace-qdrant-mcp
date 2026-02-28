/**
 * Shared setup utilities for server integration tests.
 *
 * Provides common mocks, schema, and config factory used across all
 * server-integration-*.test.ts files.
 */

import { vi } from 'vitest';
import { join } from 'node:path';
import { ServiceStatus } from '../../src/clients/grpc-types.js';
import type { ServerConfig } from '../../src/types/index.js';

// Mock the DaemonClient
export const mockDaemonClient = {
  connect: vi.fn().mockResolvedValue(undefined),
  close: vi.fn(),
  isConnected: vi.fn().mockReturnValue(true),
  registerProject: vi.fn().mockResolvedValue({
    created: true,
    project_id: 'test-project-id',
    priority: 'high',
    is_active: true,
    newly_registered: true,
  }),
  deprioritizeProject: vi.fn().mockResolvedValue({
    success: true,
    is_active: false,
    new_priority: 'normal',
  }),
  heartbeat: vi.fn().mockResolvedValue({ acknowledged: true }),
  healthCheck: vi.fn().mockResolvedValue({
    status: ServiceStatus.SERVICE_STATUS_HEALTHY,
    components: [],
  }),
  ingestText: vi.fn().mockResolvedValue({
    documentIds: ['doc-123'],
  }),
  embedText: vi.fn().mockResolvedValue({
    denseEmbedding: new Array(384).fill(0.1),
    sparseIndices: [1, 2, 3],
    sparseValues: [0.5, 0.3, 0.2],
  }),
};

// Mock the Qdrant client
export const mockQdrantClient = {
  search: vi.fn().mockResolvedValue([
    {
      id: 'result-1',
      score: 0.95,
      payload: { content: 'Test content', title: 'Test Doc' },
    },
  ]),
  retrieve: vi.fn().mockResolvedValue([
    {
      id: 'doc-1',
      payload: { content: 'Retrieved content', title: 'Doc Title' },
    },
  ]),
  scroll: vi.fn().mockResolvedValue({
    points: [
      {
        id: 'scrolled-1',
        payload: { content: 'Scrolled content' },
      },
    ],
    next_page_offset: null,
  }),
  upsert: vi.fn().mockResolvedValue({ status: 'completed' }),
  delete: vi.fn().mockResolvedValue({ status: 'completed' }),
  getCollections: vi.fn().mockResolvedValue({ collections: [{ name: 'projects' }] }),
};

// Test database schema (matching daemon's watch_folders + unified_queue tables)
export const TEST_SCHEMA = `
CREATE TABLE IF NOT EXISTS watch_folders (
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    parent_watch_id TEXT,
    submodule_path TEXT,
    git_remote_url TEXT,
    remote_hash TEXT,
    disambiguation_path TEXT,
    is_active INTEGER DEFAULT 0,
    last_activity_at TEXT,
    library_mode TEXT,
    follow_symlinks INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    cleanup_on_disable INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_scan TEXT
);

CREATE TABLE IF NOT EXISTS tracked_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    branch TEXT,
    file_type TEXT,
    language TEXT,
    file_mtime TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    chunking_method TEXT,
    lsp_status TEXT DEFAULT 'none',
    treesitter_status TEXT DEFAULT 'none',
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    collection TEXT NOT NULL DEFAULT 'projects',
    extension TEXT,
    is_test INTEGER DEFAULT 0,
    base_point TEXT,
    relative_path TEXT,
    component TEXT,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, file_path, branch)
);

CREATE TABLE IF NOT EXISTS project_components (
    component_id TEXT PRIMARY KEY,
    watch_folder_id TEXT NOT NULL,
    component_name TEXT NOT NULL,
    base_path TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'auto',
    patterns TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, component_name)
);

CREATE TABLE IF NOT EXISTS unified_queue (
    queue_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    item_type TEXT NOT NULL,
    op TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    branch TEXT,
    payload_json TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_error TEXT,
    leased_by TEXT,
    lease_expires_at TEXT
);
`;

export function createTestConfig(tempDir: string): ServerConfig {
  return {
    database: {
      path: join(tempDir, 'state.db'),
    },
    qdrant: {
      url: 'http://localhost:6333',
      timeout: 5000,
    },
    daemon: {
      grpcPort: 50051,
      queuePollIntervalMs: 1000,
      queueBatchSize: 10,
    },
    watching: {
      patterns: ['*.ts'],
      ignorePatterns: ['node_modules/*'],
    },
    collections: {
      rulesCollectionName: 'rules',
    },
    environment: {},
  };
}
