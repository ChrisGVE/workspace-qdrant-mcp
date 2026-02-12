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

interface NativeAddon {
  calculateProjectId(projectRoot: string, gitRemote: string | null): string;
  calculateProjectIdWithDisambiguation(projectRoot: string, gitRemote: string | null, disambiguationPath: string | null): string;
  normalizeGitUrl(url: string): string;
  detectGitRemote(projectRoot: string): string | null;
  calculateTenantId(projectRoot: string): string;
  generateIdempotencyKey(itemType: string, op: string, tenantId: string, collection: string, payloadJson: string): string | null;
  computeContentHash(content: string): string;
  tokenize(text: string): string[];
  collectionProjects(): string;
  collectionLibraries(): string;
  collectionMemory(): string;
  defaultQdrantUrl(): string;
  defaultGrpcPort(): number;
  defaultBranch(): string;
  isValidItemType(s: string): boolean;
  isValidQueueOperation(s: string): boolean;
  isValidQueueStatus(s: string): boolean;
  isValidOperationForType(itemType: string, op: string): boolean;
}

let addon: NativeAddon | null = null;

function loadAddon(): NativeAddon {
  if (addon) return addon;
  try {
    addon = require(NATIVE_ADDON_PATH) as NativeAddon;
    return addon;
  } catch (err) {
    throw new Error(
      `Failed to load wqm-common-node native addon from ${NATIVE_ADDON_PATH}. ` +
      `Build it with: cd src/rust/common-node && napi build --release --platform. ` +
      `Original error: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// Re-export functions
export function calculateProjectId(projectRoot: string, gitRemote: string | null): string {
  return loadAddon().calculateProjectId(projectRoot, gitRemote);
}

export function calculateProjectIdWithDisambiguation(
  projectRoot: string,
  gitRemote: string | null,
  disambiguationPath: string | null,
): string {
  return loadAddon().calculateProjectIdWithDisambiguation(projectRoot, gitRemote, disambiguationPath);
}

export function normalizeGitUrl(url: string): string {
  return loadAddon().normalizeGitUrl(url);
}

export function detectGitRemote(projectRoot: string): string | null {
  return loadAddon().detectGitRemote(projectRoot);
}

export function calculateTenantId(projectRoot: string): string {
  return loadAddon().calculateTenantId(projectRoot);
}

export function generateIdempotencyKey(
  itemType: string,
  op: string,
  tenantId: string,
  collection: string,
  payloadJson: string,
): string | null {
  return loadAddon().generateIdempotencyKey(itemType, op, tenantId, collection, payloadJson);
}

export function computeContentHash(content: string): string {
  return loadAddon().computeContentHash(content);
}

export function tokenize(text: string): string[] {
  return loadAddon().tokenize(text);
}

export function isValidItemType(s: string): boolean {
  return loadAddon().isValidItemType(s);
}

export function isValidQueueOperation(s: string): boolean {
  return loadAddon().isValidQueueOperation(s);
}

export function isValidQueueStatus(s: string): boolean {
  return loadAddon().isValidQueueStatus(s);
}

export function isValidOperationForType(itemType: string, op: string): boolean {
  return loadAddon().isValidOperationForType(itemType, op);
}

// Eagerly resolve constants (called once on import)
export const COLLECTION_PROJECTS = loadAddon().collectionProjects();
export const COLLECTION_LIBRARIES = loadAddon().collectionLibraries();
export const COLLECTION_MEMORY = loadAddon().collectionMemory();
export const DEFAULT_QDRANT_URL = loadAddon().defaultQdrantUrl();
export const DEFAULT_GRPC_PORT = loadAddon().defaultGrpcPort();
export const DEFAULT_BRANCH = loadAddon().defaultBranch();
