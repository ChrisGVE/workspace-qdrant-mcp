/**
 * Pre-enqueue URL validation tests (T5 — F-022).
 *
 * Exercises `validateUrlInput` directly. Full SSRF policy is enforced
 * daemon-side; these tests only cover the fast-fail input gate.
 */

import { describe, it, expect } from 'vitest';
import { validateUrlInput } from '../../src/store-handlers.js';

describe('validateUrlInput', () => {
  it('accepts a well-formed https URL', () => {
    const r = validateUrlInput('https://example.com/page');
    expect(r.ok).toBe(true);
  });

  it('accepts a well-formed http URL', () => {
    const r = validateUrlInput('http://example.com/');
    expect(r.ok).toBe(true);
  });

  it('rejects empty string', () => {
    const r = validateUrlInput('');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toMatch(/required/i);
  });

  it('rejects whitespace-only input', () => {
    const r = validateUrlInput('   ');
    expect(r.ok).toBe(false);
  });

  it('rejects non-string input', () => {
    const r = validateUrlInput(undefined);
    expect(r.ok).toBe(false);
  });

  it('rejects malformed URL', () => {
    const r = validateUrlInput('not a url at all');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toMatch(/malformed/i);
  });

  it('rejects file:// scheme', () => {
    const r = validateUrlInput('file:///etc/passwd');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toMatch(/http/i);
  });

  it('rejects data: scheme', () => {
    const r = validateUrlInput('data:text/plain,hello');
    expect(r.ok).toBe(false);
  });

  it('rejects ftp:// scheme', () => {
    const r = validateUrlInput('ftp://example.com/');
    expect(r.ok).toBe(false);
  });

  it('rejects javascript: scheme', () => {
    const r = validateUrlInput('javascript:alert(1)');
    expect(r.ok).toBe(false);
  });

  it('accepts URL with path and query', () => {
    const r = validateUrlInput('https://api.example.com/v1/items?id=5');
    expect(r.ok).toBe(true);
  });

  it('accepts URL with port', () => {
    const r = validateUrlInput('https://example.com:8443/');
    expect(r.ok).toBe(true);
  });

  it('trims input before parsing', () => {
    const r = validateUrlInput('  https://example.com  ');
    expect(r.ok).toBe(true);
  });
});
