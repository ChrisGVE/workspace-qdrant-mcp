import { describe, it, expect } from 'vitest';
import { DEFAULT_CONFIG } from '../../src/types/config.js';
import type { ServerConfig } from '../../src/types/config.js';

describe('Generated DEFAULT_CONFIG', () => {
  it('should satisfy the ServerConfig type', () => {
    const config: ServerConfig = DEFAULT_CONFIG;
    expect(config).toBeDefined();
  });

  it('should have qdrant defaults matching canonical YAML', () => {
    expect(DEFAULT_CONFIG.qdrant.url).toBe('http://localhost:6333');
    // YAML: 30s → 30000ms
    expect(DEFAULT_CONFIG.qdrant.timeout).toBe(30000);
  });

  it('should have daemon/gRPC defaults matching canonical YAML', () => {
    expect(DEFAULT_CONFIG.daemon.grpcPort).toBe(50051);
    expect(DEFAULT_CONFIG.daemon.queueBatchSize).toBe(10);
    expect(typeof DEFAULT_CONFIG.daemon.queuePollIntervalMs).toBe('number');
  });

  it('should have memory limits matching canonical YAML', () => {
    expect(DEFAULT_CONFIG.memory?.limits.maxLabelLength).toBe(15);
    expect(DEFAULT_CONFIG.memory?.limits.maxTitleLength).toBe(50);
    expect(DEFAULT_CONFIG.memory?.limits.maxTagLength).toBe(20);
    expect(DEFAULT_CONFIG.memory?.limits.maxTagsPerRule).toBe(5);
  });

  it('should have collections defaults matching canonical YAML', () => {
    expect(DEFAULT_CONFIG.collections.memoryCollectionName).toBe('memory');
  });

  it('should have database path as platform-specific default', () => {
    expect(DEFAULT_CONFIG.database.path).toBe('~/.workspace-qdrant/state.db');
  });

  it('should have watching patterns and ignorePatterns as arrays', () => {
    expect(Array.isArray(DEFAULT_CONFIG.watching.patterns)).toBe(true);
    expect(DEFAULT_CONFIG.watching.patterns.length).toBeGreaterThan(0);
    expect(Array.isArray(DEFAULT_CONFIG.watching.ignorePatterns)).toBe(true);
    expect(DEFAULT_CONFIG.watching.ignorePatterns.length).toBeGreaterThan(0);
  });

  it('should include key ignore patterns from YAML exclude_directories', () => {
    const ignores = DEFAULT_CONFIG.watching.ignorePatterns;
    expect(ignores).toContain('node_modules/*');
    expect(ignores).toContain('.git/*');
    expect(ignores).toContain('__pycache__/*');
    expect(ignores).toContain('target/*');
  });

  it('should include key ignore patterns from YAML exclude_patterns', () => {
    const ignores = DEFAULT_CONFIG.watching.ignorePatterns;
    expect(ignores).toContain('*.pyc');
    expect(ignores).toContain('*.class');
  });

  it('should have empty environment config', () => {
    expect(DEFAULT_CONFIG.environment).toEqual({});
  });
});
