/**
 * Integration tests for the list tool — filtering and component features.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, realpathSync } from 'node:fs';
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

  describe('List Tool Integration — Filters and Components', () => {
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

    it('should filter by file extension', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'flat', extension: 'toml', projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.listing).toContain('Cargo.toml');
      expect(data.listing).not.toContain('main.rs');
    });

    it('should exclude test files when includeTests is false', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'flat', includeTests: false, projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      expect(data.listing).not.toContain('test_main.rs');
      expect(data.listing).toContain('main.rs');
    });

    it('should include component summaries when workspace file exists', async () => {
      const projectPath = realpathSync(join(tempDir, 'list-project'));
      writeFileSync(
        join(projectPath, 'Cargo.toml'),
        '[workspace]\nresolver = "2"\nmembers = [\n    "src",\n    "tests",\n]\n',
      );

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
      expect(data.stats.components).toBeDefined();
      expect(data.stats.components.length).toBeGreaterThan(0);

      const componentIds = data.stats.components.map((c: { id: string }) => c.id);
      expect(componentIds).toContain('src');
      expect(componentIds).toContain('tests');
    });

    it('should filter files by component', async () => {
      const projectPath = realpathSync(join(tempDir, 'list-project'));
      writeFileSync(
        join(projectPath, 'Cargo.toml'),
        '[workspace]\nmembers = ["src", "tests"]\n',
      );

      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'list',
          arguments: { format: 'flat', component: 'tests', projectId: LIST_TENANT },
        },
      });

      const data = JSON.parse(result.content[0].text);
      // Should only include files under tests/
      expect(data.listing).toContain('test_main.rs');
      expect(data.listing).not.toContain('src/main.rs');
      expect(data.listing).not.toContain('README.md');
    });

    it('should read components from SQLite project_components table', async () => {
      const db = new Database(join(tempDir, 'state.db'));
      db.prepare(
        `INSERT INTO project_components (component_id, watch_folder_id, component_name, base_path, source, patterns, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run('watch-list:daemon', WATCH_ID, 'daemon', 'src', 'cargo', '["src/**"]', NOW, NOW);
      db.prepare(
        `INSERT INTO project_components (component_id, watch_folder_id, component_name, base_path, source, patterns, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run('watch-list:test-suite', WATCH_ID, 'test-suite', 'tests', 'cargo', '["tests/**"]', NOW, NOW);
      db.close();

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
      expect(data.stats.components).toBeDefined();
      const componentIds = data.stats.components.map((c: { id: string }) => c.id);
      // Should have the SQLite-persisted components
      expect(componentIds).toContain('daemon');
      expect(componentIds).toContain('test-suite');
      // Source should be from SQLite, not filesystem
      const daemon = data.stats.components.find((c: { id: string }) => c.id === 'daemon');
      expect(daemon.source).toBe('cargo');
    });
  });
});
