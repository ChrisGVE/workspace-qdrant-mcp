/**
 * Cross-language port drift guard.
 *
 * Asserts the TypeScript `DEFAULT_CONFIG` (auto-generated from
 * `assets/default_configuration.yaml`) actually matches the YAML's current
 * values, so that nobody hand-edited `generated-defaults.ts` to diverge.
 *
 * This is the TypeScript half of the cross-language drift guard. The Rust
 * half lives in `src/rust/common/src/constants.rs` (test_port_matches_yaml),
 * which asserts `DEFAULT_GRPC_PORT` matches the same YAML field.
 *
 * Failure means one of:
 *   - YAML was changed but `npm run generate:config` was not re-run.
 *   - Someone hand-edited `src/types/generated-defaults.ts`.
 *   - The YAML schema changed and the generator script needs updating.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, it, expect } from 'vitest';
import { parse as parseYaml } from 'yaml';

import { DEFAULT_CONFIG } from '../../src/types/generated-defaults.js';
import mcpPublicConfig from '../../src/constants/mcp-public-config.json' with { type: 'json' };
import {
  DEFAULT_HTTP_HOST,
  DEFAULT_HTTP_PATH,
  DEFAULT_HTTP_PORT,
} from '../../src/server-types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
// tests/admin/ → mcp-server/ → typescript/ → src/ → repo root
const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..', '..');
const YAML_PATH = resolve(REPO_ROOT, 'assets', 'default_configuration.yaml');

function loadYaml(): Record<string, unknown> {
  const text = readFileSync(YAML_PATH, 'utf8');
  return parseYaml(text) as Record<string, unknown>;
}

describe('port + URL drift guard (YAML ↔ DEFAULT_CONFIG)', () => {
  const yaml = loadYaml();

  it('grpc.port in YAML matches DEFAULT_CONFIG.daemon.grpcPort', () => {
    const grpc = yaml['grpc'] as Record<string, unknown> | undefined;
    expect(grpc, 'YAML missing top-level `grpc:` block').toBeDefined();
    expect(grpc?.['port']).toBe(DEFAULT_CONFIG.daemon.grpcPort);
  });

  it('qdrant.url in YAML matches DEFAULT_CONFIG.qdrant.url', () => {
    const qdrant = yaml['qdrant'] as Record<string, unknown> | undefined;
    expect(qdrant, 'YAML missing top-level `qdrant:` block').toBeDefined();
    expect(qdrant?.['url']).toBe(DEFAULT_CONFIG.qdrant.url);
  });

  it('DEFAULT_CONFIG.daemon.grpcPort is a sensible TCP port', () => {
    expect(DEFAULT_CONFIG.daemon.grpcPort).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.daemon.grpcPort).toBeLessThanOrEqual(65535);
    expect(Number.isInteger(DEFAULT_CONFIG.daemon.grpcPort)).toBe(true);
  });

  it('DEFAULT_CONFIG.qdrant.url parses as a valid URL', () => {
    expect(() => new URL(DEFAULT_CONFIG.qdrant.url)).not.toThrow();
  });
});

describe('MCP HTTP defaults drift guard (JSON ↔ server-types.ts)', () => {
  // Sanity: the server-types exports must match the JSON they derive from.
  // (Catches a manual edit to server-types.ts that bypasses the JSON.)
  it('DEFAULT_HTTP_HOST matches mcp-public-config.json http.host', () => {
    expect(DEFAULT_HTTP_HOST).toBe(mcpPublicConfig.http.host);
  });
  it('DEFAULT_HTTP_PORT matches mcp-public-config.json http.port', () => {
    expect(DEFAULT_HTTP_PORT).toBe(mcpPublicConfig.http.port);
  });
  it('DEFAULT_HTTP_PATH matches mcp-public-config.json http.path', () => {
    expect(DEFAULT_HTTP_PATH).toBe(mcpPublicConfig.http.path);
  });
  it('http.port is a sensible TCP port', () => {
    expect(mcpPublicConfig.http.port).toBeGreaterThan(0);
    expect(mcpPublicConfig.http.port).toBeLessThanOrEqual(65535);
    expect(Number.isInteger(mcpPublicConfig.http.port)).toBe(true);
  });
  it('http.path starts with "/"', () => {
    expect(mcpPublicConfig.http.path.startsWith('/')).toBe(true);
  });
});
