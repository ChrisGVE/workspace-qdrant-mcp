/**
 * Tests for canonicalizeHostPath — the cross-namespace path folding used by
 * project detection so a host CWD matches a daemon-stored project root.
 */

import { describe, it, expect } from 'vitest';

import { canonicalizeHostPath } from '../../src/clients/project-queries.js';

describe('canonicalizeHostPath', () => {
  it('folds Windows, Docker Desktop, WSL and MSYS forms of drive C to one path', () => {
    const expected = '/c/Users/alber/repo';
    expect(canonicalizeHostPath('C:\\Users\\alber\\repo')).toBe(expected);
    expect(canonicalizeHostPath('C:/Users/alber/repo')).toBe(expected);
    expect(canonicalizeHostPath('/run/desktop/mnt/host/c/Users/alber/repo')).toBe(expected);
    expect(canonicalizeHostPath('/mnt/c/Users/alber/repo')).toBe(expected);
    expect(canonicalizeHostPath('/c/Users/alber/repo')).toBe(expected);
  });

  it('leaves native POSIX paths unchanged', () => {
    expect(canonicalizeHostPath('/home/user/project')).toBe('/home/user/project');
    expect(canonicalizeHostPath('/test/project')).toBe('/test/project');
  });

  it('does not treat multi-letter /mnt or Docker subdirs as drive mounts', () => {
    expect(canonicalizeHostPath('/mnt/data/project')).toBe('/mnt/data/project');
    expect(canonicalizeHostPath('/run/desktop/mnt/host/config')).toBe(
      '/run/desktop/mnt/host/config'
    );
  });

  it('normalizes duplicate and trailing slashes', () => {
    expect(canonicalizeHostPath('C:\\Users\\\\alber\\repo\\')).toBe('/c/Users/alber/repo');
    expect(canonicalizeHostPath('/c/Users/alber/repo/')).toBe('/c/Users/alber/repo');
  });

  it('handles a bare drive root', () => {
    expect(canonicalizeHostPath('C:\\')).toBe('/c');
    expect(canonicalizeHostPath('C:')).toBe('/c');
  });

  it('keeps the filesystem root intact', () => {
    expect(canonicalizeHostPath('/')).toBe('/');
  });
});
