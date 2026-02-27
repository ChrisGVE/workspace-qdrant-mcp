/**
 * Integration tests for the list tool — format and basic display options.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../../src/server.js';
import type { ServerConfig } from '../../src/types/index.js';
import { mockDaemonClient, mockQdrantClient, TEST_SCHEMA, createTestConfig } from './shared-setup.js';

vi.mock('../../src/clients/daemon-client.js', () => ({
  DaemonClient: vi.fn().mockImplementation(() => mockDaemonClient),
}));

vi.mock('@modelcontextprotocol/sdk/server/index.js', () => ({
  Server: vi.fn().mockImplementation(() => ({
    connect: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
    setRequestHandler: vi.fn(),
    onerror: null,
    onclose: null,
  })),
}));

vi.mock('@modelcontextprotocol/sdk/server/stdio.js', () => ({
  StdioServerTransport: vi.fn().mockImplementation(() => ({})),
}));

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => mockQdrantClient),
}));

const NOW = '2026-02-24T12:00:00Z';
const WATCH_ID = 'watch-list';
const SUBMOD_WATCH_ID = 'watch-submod';
const LIST_TENANT = 'list-tenant';

function seedListData(dbPath: string): void {
  const db = new Database(dbPath);

  db.prepare(
    `INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
     VALUES (?, '/tmp/list-project', 'projects', 'list-tenant', 1, ?, ?)`,
  ).run(WATCH_ID, NOW, NOW);

  db.prepare(
    `INSERT INTO watch_folders (watch_id, path, collection, tenant_id, parent_watch_id, submodule_path,
     git_remote_url, is_active, created_at, updated_at)
     VALUES (?, '/tmp/list-project/vendor/lib-x', 'projects', 'list-tenant', ?, 'vendor/lib-x',
     'https://github.com/acme/lib-x.git', 1, ?, ?)`,
  ).run(SUBMOD_WATCH_ID, WATCH_ID, NOW, NOW);

  const insertFile = db.prepare(
    `INSERT INTO tracked_files
     (watch_folder_id, file_path, relative_path, branch, file_type, language,
      extension, is_test, file_mtime, file_hash, created_at, updated_at)
     VALUES (?, ?, ?, 'main', ?, ?, ?, ?, ?, 'abc123', ?, ?)`,
  );

  const files = [
    [WATCH_ID, '/tmp/list-project/src/main.rs', 'src/main.rs', 'code', 'rust', 'rs', 0],
    [WATCH_ID, '/tmp/list-project/src/lib.rs', 'src/lib.rs', 'code', 'rust', 'rs', 0],
    [WATCH_ID, '/tmp/list-project/src/utils/helpers.rs', 'src/utils/helpers.rs', 'code', 'rust', 'rs', 0],
    [WATCH_ID, '/tmp/list-project/tests/test_main.rs', 'tests/test_main.rs', 'code', 'rust', 'rs', 1],
    [WATCH_ID, '/tmp/list-project/README.md', 'README.md', 'text', null, 'md', 0],
    [WATCH_ID, '/tmp/list-project/Cargo.toml', 'Cargo.toml', 'config', null, 'toml', 0],
    [WATCH_ID, '/tmp/list-project/vendor/lib-x/src/lib.rs', 'vendor/lib-x/src/lib.rs', 'code', 'rust', 'rs', 0],
  ];

  for (const f of files) {
    insertFile.run(f[0], f[1], f[2], f[3], f[4], f[5], f[6], NOW, NOW, NOW);
  }

  db.close();
}

describe('Server Integration Tests', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-integration-test-'));
    const dbPath = join(tempDir, 'state.db');
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();
    config = createTestConfig(tempDir);
    vi.clearAllMocks();
  });

  afterEach(async () => {
    if (server) {
      await server.stop();
    }
    rmSync(tempDir, { recursive: true, force: true });
  });

  describe('List Tool Integration — Formats', () => {
    beforeEach(async () => {
      seedListData(join(tempDir, 'state.db'));

      const projectPath = join(tempDir, 'list-project');
      mkdirSync(projectPath, { recursive: true });
      mkdirSync(join(projectPath, '.git'));

      const db = new Database(join(tempDir, 'state.db'));
      const realPath = realpathSync(projectPath);
      db.prepare('UPDATE watch_folders SET path = ? WHERE watch_id = ?').run(realPath, WATCH_ID);
      db.close();

      const originalCwd = process.cwd();
      process.chdir(projectPath);

      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      process.chdir(originalCwd);
    });

    it('should list files in tree format', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'tree', projectId: LIST_TENANT },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();

      const data = JSON.parse(result.content[0].text);
      expect(data.format).toBe('tree');
      expect(data.stats.files).toBeGreaterThan(0);
      expect(data.listing).toContain('src/');
    });

    it('should list files in summary format', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'summary', projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.format).toBe('summary');
      expect(data.listing).toContain('files');
    });

    it('should list files in flat format', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'flat', projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.format).toBe('flat');
      expect(data.listing).toContain('src/main.rs');
    });

    it('should default to tree format when no format specified', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.format).toBe('tree');
    });

    it('should show submodule markers in tree view', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'tree', projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.listing).toContain('[submodule: lib-x]');
    });

    it('should respect depth limit', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'tree', depth: 1, projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      // At depth 1, nested folders should be collapsed
      expect(data.listing).not.toContain('helpers.rs');
    });

    it('should respect result limit', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'flat', limit: 2, projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.stats.truncated).toBe(true);
    });

    it('should handle empty project gracefully', async () => {
      const db = new Database(join(tempDir, 'state.db'));
      db.prepare('DELETE FROM tracked_files').run();
      db.close();

      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.stats.files).toBe(0);
    });
  });
});
