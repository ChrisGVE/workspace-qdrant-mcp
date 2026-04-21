/**
 * Unit tests for the HTTP auth/rate-limit/CORS middleware.
 */

import { describe, it, expect } from 'vitest';
import {
  constantTimeEquals,
  extractBearer,
  loadAuthConfig,
  requireAuth,
  tokenDigest,
} from '../src/auth-middleware.js';

describe('auth-middleware', () => {
  describe('extractBearer', () => {
    it('returns the token for a valid header', () => {
      expect(extractBearer('Bearer abc123')).toBe('abc123');
    });

    it('trims whitespace around the token', () => {
      expect(extractBearer('Bearer    abc123   ')).toBe('abc123');
    });

    it('returns null for missing header', () => {
      expect(extractBearer(undefined)).toBeNull();
    });

    it('returns null for non-Bearer schemes', () => {
      expect(extractBearer('Basic dXNlcjpwYXNz')).toBeNull();
      expect(extractBearer('Token abc')).toBeNull();
    });

    it('returns null for an empty bearer value', () => {
      expect(extractBearer('Bearer   ')).toBeNull();
    });

    it('matches with trailing whitespace in the overall header', () => {
      expect(extractBearer('  Bearer abc123  ')).toBe('abc123');
    });
  });

  describe('constantTimeEquals', () => {
    it('returns true for equal strings', () => {
      expect(constantTimeEquals('abcdef', 'abcdef')).toBe(true);
    });

    it('returns false for different strings of equal length', () => {
      expect(constantTimeEquals('abcdef', 'abcdez')).toBe(false);
    });

    it('returns false for strings of different length without throwing', () => {
      expect(constantTimeEquals('short', 'longer-string')).toBe(false);
    });

    it('handles multibyte UTF-8 safely', () => {
      expect(constantTimeEquals('héllo', 'héllo')).toBe(true);
      expect(constantTimeEquals('héllo', 'hello')).toBe(false);
    });
  });

  describe('tokenDigest', () => {
    it('returns 8 hex chars', () => {
      const digest = tokenDigest('some-token-value');
      expect(digest).toMatch(/^[0-9a-f]{8}$/);
    });

    it('is deterministic', () => {
      expect(tokenDigest('x')).toBe(tokenDigest('x'));
    });

    it('differs for distinct tokens', () => {
      expect(tokenDigest('one')).not.toBe(tokenDigest('two'));
    });
  });

  describe('loadAuthConfig', () => {
    it('defaults to 100 rpm, no CORS, null token when env is empty', () => {
      const cfg = loadAuthConfig({});
      expect(cfg.token).toBeNull();
      expect(cfg.rateLimitPerMin).toBe(100);
      expect(cfg.corsOrigins).toEqual([]);
    });

    it('reads all three env vars', () => {
      const cfg = loadAuthConfig({
        MCP_HTTP_TOKEN: 'secret-token-value-1234567890ab',
        MCP_HTTP_RATE_LIMIT: '250',
        MCP_HTTP_CORS_ORIGINS: 'https://a.example, https://b.example',
      });
      expect(cfg.token).toBe('secret-token-value-1234567890ab');
      expect(cfg.rateLimitPerMin).toBe(250);
      expect(cfg.corsOrigins).toEqual(['https://a.example', 'https://b.example']);
    });

    it('throws on a non-integer rate limit', () => {
      expect(() => loadAuthConfig({ MCP_HTTP_RATE_LIMIT: 'abc' })).toThrow(/MCP_HTTP_RATE_LIMIT/);
    });

    it('throws on a zero rate limit', () => {
      expect(() => loadAuthConfig({ MCP_HTTP_RATE_LIMIT: '0' })).toThrow(/MCP_HTTP_RATE_LIMIT/);
    });
  });

  describe('requireAuth', () => {
    it('accepts a token of sufficient length', () => {
      expect(() =>
        requireAuth({ token: 'a'.repeat(32), rateLimitPerMin: 100, corsOrigins: [] })
      ).not.toThrow();
    });

    it('throws on a null token', () => {
      expect(() => requireAuth({ token: null, rateLimitPerMin: 100, corsOrigins: [] })).toThrow(
        /MCP_HTTP_TOKEN is required/
      );
    });

    it('throws on an empty token', () => {
      expect(() => requireAuth({ token: '', rateLimitPerMin: 100, corsOrigins: [] })).toThrow(
        /MCP_HTTP_TOKEN is required/
      );
    });

    it('throws on a too-short token', () => {
      expect(() => requireAuth({ token: 'short', rateLimitPerMin: 100, corsOrigins: [] })).toThrow(
        /at least 16/
      );
    });
  });
});
