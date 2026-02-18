/**
 * Tests for base point identity computation.
 * Includes cross-language parity tests with known Rust test vectors.
 */

import { describe, it, expect } from 'vitest';
import {
  normalizePathForId,
  computeBasePoint,
  computePointId,
} from '../../src/utils/base-point.js';

describe('normalizePathForId', () => {
  it('should pass through unix paths unchanged', () => {
    expect(normalizePathForId('src/main.rs')).toBe('src/main.rs');
  });

  it('should convert backslashes to forward slashes', () => {
    expect(normalizePathForId('src\\main.rs')).toBe('src/main.rs');
  });

  it('should strip trailing slashes', () => {
    expect(normalizePathForId('src/dir/')).toBe('src/dir');
  });

  it('should handle windows-style paths', () => {
    expect(normalizePathForId('src\\dir\\file.rs')).toBe('src/dir/file.rs');
  });
});

describe('computeBasePoint', () => {
  it('should be deterministic', () => {
    const bp1 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'deadbeef');
    const bp2 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'deadbeef');
    expect(bp1).toBe(bp2);
    expect(bp1).toHaveLength(32);
    expect(bp1).toMatch(/^[0-9a-f]{32}$/);
  });

  it('should differ with different file_hash', () => {
    const bp1 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'hash_v1');
    const bp2 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'hash_v2');
    expect(bp1).not.toBe(bp2);
  });

  it('should differ with different branch', () => {
    const bp1 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'deadbeef');
    const bp2 = computeBasePoint('tenant_abc', 'feature', 'src/main.rs', 'deadbeef');
    expect(bp1).not.toBe(bp2);
  });

  it('should differ with different path', () => {
    const bp1 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'deadbeef');
    const bp2 = computeBasePoint('tenant_abc', 'main', 'src/lib.rs', 'deadbeef');
    expect(bp1).not.toBe(bp2);
  });

  it('should differ with different tenant', () => {
    const bp1 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'deadbeef');
    const bp2 = computeBasePoint('tenant_xyz', 'main', 'src/main.rs', 'deadbeef');
    expect(bp1).not.toBe(bp2);
  });

  it('should normalize path (backslash parity)', () => {
    const bp1 = computeBasePoint('t', 'main', 'src/main.rs', 'h');
    const bp2 = computeBasePoint('t', 'main', 'src\\main.rs', 'h');
    expect(bp1).toBe(bp2);
  });
});

describe('computePointId', () => {
  it('should be deterministic', () => {
    const bp = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'deadbeef');
    const pid1 = computePointId(bp, 0);
    const pid2 = computePointId(bp, 0);
    expect(pid1).toBe(pid2);
    expect(pid1).toHaveLength(32);
  });

  it('should differ with different chunk_index', () => {
    const bp = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'deadbeef');
    const pid0 = computePointId(bp, 0);
    const pid1 = computePointId(bp, 1);
    const pid2 = computePointId(bp, 2);
    expect(pid0).not.toBe(pid1);
    expect(pid1).not.toBe(pid2);
    expect(pid0).not.toBe(pid2);
  });

  it('should differ with different base_point', () => {
    const bp1 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'hash_v1');
    const bp2 = computeBasePoint('tenant_abc', 'main', 'src/main.rs', 'hash_v2');
    const pid1 = computePointId(bp1, 0);
    const pid2 = computePointId(bp2, 0);
    expect(pid1).not.toBe(pid2);
  });
});

describe('cross-language parity with Rust', () => {
  // These test vectors are from wqm-common/src/hashing.rs
  // Input: compute_base_point("test_tenant", "main", "src/example.rs", "abc123hash")

  it('should match Rust base_point output', () => {
    const bp = computeBasePoint('test_tenant', 'main', 'src/example.rs', 'abc123hash');
    expect(bp).toBe('fb8f745c24089bb3c094002a25e4762d');
  });

  it('should match Rust point_id output (chunk 0)', () => {
    const bp = computeBasePoint('test_tenant', 'main', 'src/example.rs', 'abc123hash');
    const pid = computePointId(bp, 0);
    expect(pid).toBe('29f8fee936e7f18423f871d91da964fa');
  });
});
