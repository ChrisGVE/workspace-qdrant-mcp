/**
 * Cross-language compatibility tests for the native bridge
 *
 * These tests verify that the napi-rs native addon produces identical results
 * to the Rust wqm-common crate. Test vectors are generated from known Rust
 * outputs to ensure both implementations stay in sync.
 */

import { describe, it, expect } from 'vitest';
import {
  calculateProjectId,
  calculateProjectIdWithDisambiguation,
  normalizeGitUrl,
  generateIdempotencyKey,
  computeContentHash,
  tokenize,
  isValidItemType,
  isValidQueueOperation,
  isValidQueueStatus,
  isValidOperationForType,
  COLLECTION_PROJECTS,
  COLLECTION_LIBRARIES,
  COLLECTION_MEMORY,
  DEFAULT_QDRANT_URL,
  DEFAULT_GRPC_PORT,
  DEFAULT_BRANCH,
} from '../../src/common/native-bridge.js';

describe('native-bridge: constants', () => {
  it('should return canonical collection names', () => {
    expect(COLLECTION_PROJECTS).toBe('projects');
    expect(COLLECTION_LIBRARIES).toBe('libraries');
    expect(COLLECTION_MEMORY).toBe('memory');
  });

  it('should return default config values', () => {
    expect(DEFAULT_QDRANT_URL).toBe('http://localhost:6333');
    expect(DEFAULT_GRPC_PORT).toBe(50051);
    expect(DEFAULT_BRANCH).toBe('main');
  });
});

describe('native-bridge: git URL normalization', () => {
  it('should normalize HTTPS URL', () => {
    expect(normalizeGitUrl('https://github.com/user/repo.git')).toBe('github.com/user/repo');
  });

  it('should normalize SSH URL', () => {
    expect(normalizeGitUrl('git@github.com:user/repo.git')).toBe('github.com/user/repo');
  });

  it('should normalize HTTP URL without .git', () => {
    expect(normalizeGitUrl('http://github.com/user/repo')).toBe('github.com/user/repo');
  });

  it('should be case-insensitive', () => {
    expect(normalizeGitUrl('https://GitHub.COM/User/Repo.git')).toBe('github.com/user/repo');
  });

  it('should produce same result for SSH and HTTPS of same repo', () => {
    const ssh = normalizeGitUrl('git@github.com:user/repo.git');
    const https = normalizeGitUrl('https://github.com/user/repo.git');
    expect(ssh).toBe(https);
  });
});

describe('native-bridge: project ID calculation', () => {
  it('should produce 12-char hex for remote project', () => {
    const id = calculateProjectId('/home/user/project', 'https://github.com/user/repo.git');
    expect(id).toHaveLength(12);
    expect(id).toMatch(/^[0-9a-f]+$/);
  });

  it('should produce local_ prefix for non-git project', () => {
    const id = calculateProjectId('/home/user/project', null);
    expect(id).toMatch(/^local_[0-9a-f]{12}$/);
  });

  it('should produce same ID for SSH and HTTPS of same repo', () => {
    const id1 = calculateProjectId('/path1', 'https://github.com/user/repo.git');
    const id2 = calculateProjectId('/path2', 'git@github.com:user/repo.git');
    expect(id1).toBe(id2);
  });

  it('should produce different IDs with disambiguation', () => {
    const id1 = calculateProjectIdWithDisambiguation(
      '/home/user/work/project',
      'https://github.com/user/repo.git',
      'work/project',
    );
    const id2 = calculateProjectIdWithDisambiguation(
      '/home/user/personal/project',
      'https://github.com/user/repo.git',
      'personal/project',
    );
    expect(id1).not.toBe(id2);
  });
});

describe('native-bridge: idempotency key generation', () => {
  it('should produce 32-char hex key', () => {
    const key = generateIdempotencyKey('file', 'add', 'proj_abc123', 'my-project-code', '{}');
    expect(key).not.toBeNull();
    expect(key!).toHaveLength(32);
    expect(key!).toMatch(/^[0-9a-f]+$/);
  });

  it('should be deterministic (same input -> same output)', () => {
    const key1 = generateIdempotencyKey('file', 'add', 'proj_abc123', 'projects', '{"file_path":"/path/to/file.rs"}');
    const key2 = generateIdempotencyKey('file', 'add', 'proj_abc123', 'projects', '{"file_path":"/path/to/file.rs"}');
    expect(key1).toBe(key2);
  });

  it('should produce different keys for different payloads', () => {
    const key1 = generateIdempotencyKey('file', 'add', 'proj', 'projects', '{"file_path":"/a.rs"}');
    const key2 = generateIdempotencyKey('file', 'add', 'proj', 'projects', '{"file_path":"/b.rs"}');
    expect(key1).not.toBe(key2);
  });

  it('should return null for invalid type/op combinations', () => {
    // doc only supports delete and uplift, not add
    const key = generateIdempotencyKey('doc', 'add', 'proj', 'projects', '{}');
    expect(key).toBeNull();
  });

  it('should return null for invalid item type', () => {
    const key = generateIdempotencyKey('nonexistent', 'add', 'proj', 'projects', '{}');
    expect(key).toBeNull();
  });
});

describe('native-bridge: content hashing', () => {
  it('should produce 64-char SHA256 hex', () => {
    const hash = computeContentHash('hello world');
    expect(hash).toHaveLength(64);
    expect(hash).toMatch(/^[0-9a-f]+$/);
  });

  it('should be deterministic', () => {
    expect(computeContentHash('hello world')).toBe(computeContentHash('hello world'));
  });

  it('should produce different hashes for different content', () => {
    expect(computeContentHash('hello')).not.toBe(computeContentHash('world'));
  });

  // Cross-language test vector: known SHA256 of "hello world"
  it('should match known SHA256 value', () => {
    expect(computeContentHash('hello world')).toBe(
      'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9',
    );
  });
});

describe('native-bridge: tokenization', () => {
  it('should tokenize and lowercase', () => {
    const tokens = tokenize('Hello World');
    expect(tokens).toContain('hello');
    expect(tokens).toContain('world');
  });

  it('should remove stopwords', () => {
    const tokens = tokenize('Hello World, this is a test!');
    expect(tokens).toContain('hello');
    expect(tokens).toContain('world');
    expect(tokens).toContain('test');
    // Stopwords removed
    expect(tokens).not.toContain('this');
    expect(tokens).not.toContain('is');
    expect(tokens).not.toContain('a');
  });

  it('should handle code tokens', () => {
    const tokens = tokenize('fn process_file(path: &str) -> Result<()>');
    expect(tokens).toContain('fn');
    expect(tokens).toContain('process_file');
    expect(tokens).toContain('path');
    expect(tokens).toContain('result');
  });

  it('should return empty for empty input', () => {
    expect(tokenize('')).toEqual([]);
  });

  it('should return empty for all-stopwords input', () => {
    expect(tokenize('the and or but')).toEqual([]);
  });
});

describe('native-bridge: queue type validation', () => {
  it('should validate item types', () => {
    expect(isValidItemType('text')).toBe(true);
    expect(isValidItemType('file')).toBe(true);
    expect(isValidItemType('url')).toBe(true);
    expect(isValidItemType('website')).toBe(true);
    expect(isValidItemType('doc')).toBe(true);
    expect(isValidItemType('folder')).toBe(true);
    expect(isValidItemType('tenant')).toBe(true);
    expect(isValidItemType('collection')).toBe(true);
    expect(isValidItemType('invalid')).toBe(false);
    expect(isValidItemType('')).toBe(false);
  });

  it('should validate queue operations', () => {
    expect(isValidQueueOperation('add')).toBe(true);
    expect(isValidQueueOperation('update')).toBe(true);
    expect(isValidQueueOperation('delete')).toBe(true);
    expect(isValidQueueOperation('scan')).toBe(true);
    expect(isValidQueueOperation('rename')).toBe(true);
    expect(isValidQueueOperation('uplift')).toBe(true);
    expect(isValidQueueOperation('reset')).toBe(true);
    expect(isValidQueueOperation('invalid')).toBe(false);
  });

  it('should validate queue statuses', () => {
    expect(isValidQueueStatus('pending')).toBe(true);
    expect(isValidQueueStatus('in_progress')).toBe(true);
    expect(isValidQueueStatus('done')).toBe(true);
    expect(isValidQueueStatus('failed')).toBe(true);
    expect(isValidQueueStatus('invalid')).toBe(false);
  });

  it('should validate operation+type combinations', () => {
    // Valid combinations
    expect(isValidOperationForType('file', 'add')).toBe(true);
    expect(isValidOperationForType('file', 'update')).toBe(true);
    expect(isValidOperationForType('file', 'delete')).toBe(true);
    expect(isValidOperationForType('file', 'rename')).toBe(true);
    expect(isValidOperationForType('folder', 'scan')).toBe(true);
    expect(isValidOperationForType('tenant', 'delete')).toBe(true);
    expect(isValidOperationForType('collection', 'uplift')).toBe(true);

    // Invalid combinations
    expect(isValidOperationForType('file', 'scan')).toBe(false);
    expect(isValidOperationForType('doc', 'add')).toBe(false);
    expect(isValidOperationForType('collection', 'add')).toBe(false);
    expect(isValidOperationForType('folder', 'update')).toBe(false);
  });
});
