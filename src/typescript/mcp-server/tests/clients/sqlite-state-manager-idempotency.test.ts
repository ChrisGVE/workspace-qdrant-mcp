/**
 * Tests for generateIdempotencyKey and payload builder utilities
 */

import { describe, it, expect } from 'vitest';

import {
  generateIdempotencyKey,
  buildContentPayload,
  buildRulesPayload,
  buildLibraryPayload,
} from '../../src/clients/sqlite-state-manager.js';

describe('generateIdempotencyKey', () => {
  it('should generate consistent 32-character hex key', () => {
    const key = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', {
      content: 'test',
    });

    expect(key).toHaveLength(32);
    expect(key).toMatch(/^[a-f0-9]+$/);
  });

  it('should generate same key for same inputs', () => {
    const payload = { content: 'test', source: 'user' };

    const key1 = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', payload);
    const key2 = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', payload);

    expect(key1).toBe(key2);
  });

  it('should generate different keys for different inputs', () => {
    const key1 = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', {
      a: 1,
    });
    const key2 = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', {
      a: 2,
    });

    expect(key1).not.toBe(key2);
  });

  it('should generate different keys for different operations', () => {
    const payload = { content: 'test' };

    const key1 = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', payload);
    const key2 = generateIdempotencyKey('text', 'update', 'tenant1', 'collection1', payload);

    expect(key1).not.toBe(key2);
  });

  it('should sort payload keys for consistency', () => {
    const key1 = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', {
      b: 2,
      a: 1,
    });
    const key2 = generateIdempotencyKey('text', 'add', 'tenant1', 'collection1', {
      a: 1,
      b: 2,
    });

    expect(key1).toBe(key2);
  });
});

describe('payload builders', () => {
  describe('buildContentPayload', () => {
    it('should build content payload with required fields', () => {
      const payload = buildContentPayload('test content', 'user_input');

      expect(payload).toEqual({
        content: 'test content',
        source_type: 'user_input',
        main_tag: undefined,
        full_tag: undefined,
      });
    });

    it('should build content payload with optional fields', () => {
      const payload = buildContentPayload('test', 'web', 'main_tag', 'full_tag');

      expect(payload).toEqual({
        content: 'test',
        source_type: 'web',
        main_tag: 'main_tag',
        full_tag: 'full_tag',
      });
    });
  });

  describe('buildRulesPayload', () => {
    it('should build rules payload for global scope', () => {
      const payload = buildRulesPayload('prefer-uv', 'Use uv for Python packages', 'global');

      expect(payload).toEqual({
        content: 'Use uv for Python packages',
        source_type: 'rule',
        label: 'prefer-uv',
        scope: 'global',
        project_id: undefined,
      });
    });

    it('should build rules payload for project scope', () => {
      const payload = buildRulesPayload('use-pytest', 'Use pytest', 'project', 'abc123');

      expect(payload).toEqual({
        content: 'Use pytest',
        source_type: 'rule',
        label: 'use-pytest',
        scope: 'project',
        project_id: 'abc123',
      });
    });
  });

  describe('buildLibraryPayload', () => {
    it('should build library payload with required fields', () => {
      const payload = buildLibraryPayload('numpy');

      expect(payload).toEqual({
        library_name: 'numpy',
        content: undefined,
        source: undefined,
        url: undefined,
      });
    });

    it('should build library payload with optional fields', () => {
      const payload = buildLibraryPayload('numpy', 'doc content', 'web', 'https://numpy.org');

      expect(payload).toEqual({
        library_name: 'numpy',
        content: 'doc content',
        source: 'web',
        url: 'https://numpy.org',
      });
    });
  });
});
