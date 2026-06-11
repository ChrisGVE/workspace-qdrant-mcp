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

  it('folds a WSL UNC share (Windows host editing an ext4 repo) to its POSIX root', () => {
    // The daemon runs inside the distro and stores `/home/alkmimm/...`; a
    // Windows-host client reports the `\\wsl.localhost\<distro>\...` UNC view.
    const expected = '/home/alkmimm/respositorios/DOC-V2';
    expect(canonicalizeHostPath('\\\\wsl.localhost\\ubuntu-24.04\\home\\alkmimm\\respositorios\\DOC-V2')).toBe(expected);
    expect(canonicalizeHostPath('//wsl.localhost/ubuntu-24.04/home/alkmimm/respositorios/DOC-V2')).toBe(expected);
    // Legacy `\\wsl$\<distro>\...` form, and case-insensitive share host.
    expect(canonicalizeHostPath('\\\\wsl$\\Ubuntu-24.04\\home\\alkmimm\\respositorios\\DOC-V2')).toBe(expected);
    expect(canonicalizeHostPath('//WSL.localhost/ubuntu-24.04/home/alkmimm/respositorios/DOC-V2')).toBe(expected);
  });

  it('does not treat a plain POSIX dir named like the share as a WSL UNC path', () => {
    // No `wsl.localhost`/`wsl$` host segment → left untouched.
    expect(canonicalizeHostPath('/home/wsl/project')).toBe('/home/wsl/project');
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
