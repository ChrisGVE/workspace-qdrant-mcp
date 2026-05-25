/**
 * Tests for detectCurrentBranch utility.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { detectCurrentBranch } from '../../src/utils/git-branch.js';

describe('detectCurrentBranch', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'git-branch-test-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('returns the branch name from a symbolic ref HEAD', () => {
    mkdirSync(join(tempDir, '.git'));
    writeFileSync(join(tempDir, '.git', 'HEAD'), 'ref: refs/heads/main\n');

    expect(detectCurrentBranch(tempDir)).toBe('main');
  });

  it('returns the branch name for a non-main branch', () => {
    mkdirSync(join(tempDir, '.git'));
    writeFileSync(join(tempDir, '.git', 'HEAD'), 'ref: refs/heads/feature/my-feature\n');

    expect(detectCurrentBranch(tempDir)).toBe('feature/my-feature');
  });

  it('returns first 8 chars of SHA for detached HEAD', () => {
    mkdirSync(join(tempDir, '.git'));
    writeFileSync(join(tempDir, '.git', 'HEAD'), 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0\n');

    expect(detectCurrentBranch(tempDir)).toBe('a1b2c3d4');
  });

  it('returns "default" when no .git directory exists', () => {
    expect(detectCurrentBranch(tempDir)).toBe('default');
  });

  it('returns "default" when .git/HEAD file does not exist', () => {
    mkdirSync(join(tempDir, '.git'));
    // No HEAD file written

    expect(detectCurrentBranch(tempDir)).toBe('default');
  });

  it('returns "default" for unrecognised HEAD content', () => {
    mkdirSync(join(tempDir, '.git'));
    writeFileSync(join(tempDir, '.git', 'HEAD'), 'something-unexpected\n');

    expect(detectCurrentBranch(tempDir)).toBe('default');
  });

  it('detects branch from a subdirectory of the git root', () => {
    mkdirSync(join(tempDir, '.git'));
    writeFileSync(join(tempDir, '.git', 'HEAD'), 'ref: refs/heads/dev\n');
    mkdirSync(join(tempDir, 'src', 'deep'), { recursive: true });

    // Pass a subdirectory — should walk up and find the .git root
    expect(detectCurrentBranch(join(tempDir, 'src', 'deep'))).toBe('dev');
  });
});
