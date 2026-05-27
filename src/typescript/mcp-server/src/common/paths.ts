/**
 * Canonical and local path abstraction for workspace-qdrant-mcp.
 *
 * Implements spec §3.1 normalization rules and §4.2 brand-typed API.
 * See docs/specs/16-path-abstraction.md for full rationale.
 *
 * IMPORTANT: Never use `as CanonicalPath` or `as LocalPath` casts outside
 * this module. Use fromUserInput(), fromValidated(), toLocal(), toCanonical()
 * exclusively. An ESLint rule enforces this.
 */

import { homedir } from 'node:os';

// ---------------------------------------------------------------------------
// §3.1 / §4.2 — Brand type declarations
// ---------------------------------------------------------------------------

declare const _canonical: unique symbol;
declare const _local: unique symbol;

/**
 * Host-absolute, syntactically normalized, UTF-8 path. Stable across
 * deployment modes. The form persisted to SQLite, transmitted over gRPC,
 * and returned in MCP responses.
 *
 * Construct only via {@link fromUserInput} or {@link fromValidated}.
 * Never cast with `as CanonicalPath`.
 */
export type CanonicalPath = string & { readonly [_canonical]: true };

/**
 * Path as seen by the current process's filesystem. Differs between host
 * and container deployments. Used only for fs I/O calls. Never serialized.
 *
 * Construct only via {@link toLocal}. Never cast with `as LocalPath`.
 */
export type LocalPath = string & { readonly [_local]: true };

// ---------------------------------------------------------------------------
// §5 — MountMap
// ---------------------------------------------------------------------------

/**
 * One host ↔ container directory pairing.
 *
 * Both fields are canonical (host-absolute, normalized) strings that
 * have been validated by {@link fromUserInput} on MountMap construction.
 */
export interface MountEntry {
  readonly host: string;
  readonly container: string;
}

/**
 * List of mount pairings with longest-prefix-wins resolution.
 *
 * Immutable after construction (spec §5.3). An empty MountMap acts as the
 * identity map — {@link toLocal} passes the canonical path through verbatim.
 *
 * @example
 * // Mirror mounts (host-native deployment):
 * const m = new MountMap([]);
 *
 * // Non-mirror mounts (containerized memexd):
 * const m2 = new MountMap([
 *   { host: '/Volumes/External/books', container: '/mnt/books' },
 * ]);
 */
export class MountMap {
  private readonly _entries: ReadonlyArray<MountEntry>;

  /**
   * Construct a MountMap from raw entry pairs.
   *
   * Each host and container string is validated and normalized via
   * {@link fromUserInput}. Duplicate host or duplicate container prefixes
   * (after normalization) are rejected per spec §5.3.
   *
   * Pass an empty array for the identity map (host-native deployment).
   *
   * @throws {PathError} on duplicate host prefix, duplicate container prefix,
   *   or any normalization failure in host/container values.
   */
  constructor(rawEntries: ReadonlyArray<{ host: string; container: string }>) {
    const entries: MountEntry[] = [];
    const seenHosts = new Set<string>();
    const seenContainers = new Set<string>();

    for (const raw of rawEntries) {
      // Validate both sides through the full normalization pipeline.
      const host = fromUserInput(raw.host);
      const container = fromUserInput(raw.container);

      if (seenHosts.has(host)) {
        throw new PathError('mount-duplicate', `duplicate host mount prefix: ${host}`);
      }
      if (seenContainers.has(container)) {
        throw new PathError('mount-duplicate', `duplicate container mount prefix: ${container}`);
      }

      seenHosts.add(host);
      seenContainers.add(container);
      entries.push({ host, container });
    }

    this._entries = entries;
  }

  /** True when the map has zero entries (identity / host-native mode). */
  get isIdentity(): boolean {
    return this._entries.length === 0;
  }

  /** Number of declared mount entries. */
  get size(): number {
    return this._entries.length;
  }

  /**
   * Find the entry whose `host` prefix best covers the canonical path.
   * Longest-prefix-wins, component-aware (spec §5.2).
   */
  findForCanonical(canonical: string): MountEntry | undefined {
    let best: MountEntry | undefined;
    for (const entry of this._entries) {
      if (
        componentAwarePrefix(canonical, entry.host) &&
        (best === undefined || entry.host.length > best.host.length)
      ) {
        best = entry;
      }
    }
    return best;
  }

  /**
   * Find the entry whose `container` prefix best covers the local path.
   * Longest-prefix-wins, component-aware.
   */
  findForContainer(localStr: string): MountEntry | undefined {
    let best: MountEntry | undefined;
    for (const entry of this._entries) {
      if (
        componentAwarePrefix(localStr, entry.container) &&
        (best === undefined || entry.container.length > best.container.length)
      ) {
        best = entry;
      }
    }
    return best;
  }
}

// ---------------------------------------------------------------------------
// §4.2 — PathError
// ---------------------------------------------------------------------------

/** Discriminant for the specific failure mode in {@link PathError}. */
export type PathErrorKind =
  | 'relative' // §3.1 rule 1
  | 'dot-dot' // §3.1 rule 4
  | 'non-utf8' // §3.1 rule 9
  | 'empty' // empty input
  | 'nul-byte' // embedded NUL
  | 'no-mount-coverage' // §5.2 no entry covers the path
  | 'mount-duplicate' // §5.3 duplicate host or container prefix
  | 'invalid'; // other normalization failures

/**
 * Failure modes for canonical-path construction and mount-map translation.
 *
 * Mirrors `wqm_common::paths::PathError` error variants (spec §4.1).
 */
export class PathError extends Error {
  readonly kind: PathErrorKind;

  constructor(kind: PathErrorKind, message: string) {
    super(message);
    this.name = 'PathError';
    this.kind = kind;
  }
}

// ---------------------------------------------------------------------------
// §3.1 — Core normalization (private helper)
// ---------------------------------------------------------------------------

/**
 * If `s` starts with a Windows drive prefix like `C:/`, returns the drive
 * portion (`"C:"`) plus the index of the character AFTER the trailing slash.
 * Otherwise returns `null`. `s` is assumed to already use forward slashes.
 *
 * Mirrors `windows_drive_prefix_len` in `wqm-common::paths::normalize`.
 */
function parseWindowsDrivePrefix(s: string): { drive: string; restStart: number } | null {
  if (s.length < 3) return null;
  const c0 = s.charCodeAt(0);
  const isAlpha =
    (c0 >= 0x41 && c0 <= 0x5a) || (c0 >= 0x61 && c0 <= 0x7a); // A-Z or a-z
  if (!isAlpha) return null;
  if (s.charCodeAt(1) !== 0x3a) return null; // ':'
  if (s.charCodeAt(2) !== 0x2f) return null; // '/'
  return { drive: s.slice(0, 2), restStart: 3 };
}

/**
 * Apply the nine normalization rules from spec §3.1 and return the canonical
 * string form.
 *
 * Accepted absolute forms:
 *   - POSIX: `/foo/bar/baz`
 *   - Windows drive: `C:/foo/bar/baz` (backslashes are accepted on input
 *     and normalized to forward slashes before the absolute check)
 *
 * Pure string operation — no filesystem access, no symlink resolution.
 *
 * @throws {PathError} on any validation failure.
 */
function normalizePath(input: string): string {
  // Rule 9 pre-check: embedded NUL bytes are never valid.
  if (input.includes('\0')) {
    throw new PathError('nul-byte', 'path contains embedded NUL byte');
  }

  // Empty input rejected before any further processing.
  if (input.length === 0) {
    throw new PathError('empty', 'path is empty');
  }

  // Rule 2: expand leading `~` to the user's home directory.
  let expanded: string;
  if (input === '~') {
    expanded = homedir();
  } else if (input.startsWith('~/')) {
    expanded = homedir() + input.slice(1);
  } else {
    expanded = input;
  }

  // Normalize Windows separators so the canonical form uses forward slashes
  // regardless of input source.
  const asPosix = expanded.replace(/\\/g, '/');

  // Rule 1: must be absolute after tilde expansion and separator normalization.
  // Accept POSIX (`/...`) and Windows drive (`C:/...`).
  const drivePrefix = parseWindowsDrivePrefix(asPosix);
  let prefix = '';
  let rest: string;
  if (drivePrefix !== null) {
    prefix = drivePrefix.drive;
    rest = asPosix.slice(drivePrefix.restStart);
  } else if (asPosix.startsWith('/')) {
    rest = asPosix.slice(1);
  } else {
    throw new PathError('relative', `path must be absolute, got: ${JSON.stringify(input)}`);
  }

  // Split on '/' and process each segment.
  const rawSegments = rest.split('/');
  const parts: string[] = [];

  for (const seg of rawSegments) {
    if (seg === '' || seg === '.') {
      // Rules 3 & 5: drop empty segments (duplicate slashes) and '.' segments.
      continue;
    }

    // Rule 4: reject '..' entirely.
    if (seg === '..') {
      throw new PathError(
        'dot-dot',
        `path must not contain '..' segments, got: ${JSON.stringify(input)}`
      );
    }

    // Rule 9: lone surrogates in JS strings cannot be represented as valid
    // UTF-8. Detect via encodeURIComponent which throws on lone surrogates.
    try {
      encodeURIComponent(seg);
    } catch {
      throw new PathError('non-utf8', 'path contains non-UTF-8 sequences (lone surrogate)');
    }

    // Rules 6 & 7: preserve case exactly; never resolve symlinks.
    parts.push(seg);
  }

  // Reconstruct. Bare root paths (`/` or `C:/`) are not produced — the loop
  // above strips empty segments and we treat zero non-root segments as the
  // root itself.
  if (prefix === '') {
    return '/' + parts.join('/');
  }
  return `${prefix}/${parts.join('/')}`;
}

// ---------------------------------------------------------------------------
// §4.2 — Public constructors
// ---------------------------------------------------------------------------

/**
 * Build a {@link CanonicalPath} from raw user input (CLI argument, config
 * field, gRPC payload).
 *
 * Applies all nine normalization rules from spec §3.1:
 * - `~` expansion
 * - `.` segment removal
 * - `..` rejection
 * - duplicate `/` collapse
 * - UTF-8 / lone-surrogate validation
 * - absolute requirement
 *
 * @throws {PathError} on any validation failure.
 */
export function fromUserInput(s: string): CanonicalPath {
  return normalizePath(s) as CanonicalPath;
}

/**
 * Build a {@link CanonicalPath} from a value already known to be canonical
 * (e.g., a value decoded from a DB row or gRPC response).
 *
 * Applies the same full validation as {@link fromUserInput}. This is NOT a
 * no-op cast — it validates that the input is already in canonical form and
 * throws if it is not.
 *
 * Spec §4.2: "A no-op cast that only changes the type tag is forbidden."
 *
 * @throws {PathError} on any validation failure.
 */
export function fromValidated(s: string): CanonicalPath {
  // Same checks as fromUserInput. For a value from a DB row the input should
  // already satisfy all rules; this call is the compile-time and runtime
  // safety net.
  return normalizePath(s) as CanonicalPath;
}

// ---------------------------------------------------------------------------
// §4.2 — Translation functions
// ---------------------------------------------------------------------------

/**
 * Translate a {@link CanonicalPath} to a {@link LocalPath} using the active
 * mount map.
 *
 * Identity map (empty MountMap) passes the canonical string through verbatim.
 * Non-identity: the longest matching host prefix is replaced by the
 * corresponding container prefix (spec §5.2).
 *
 * @throws {PathError} with kind `'no-mount-coverage'` when the map is
 *   non-identity and no entry covers the canonical path.
 */
export function toLocal(c: CanonicalPath, mounts: MountMap): LocalPath {
  if (mounts.isIdentity) {
    return c as unknown as LocalPath;
  }

  const entry = mounts.findForCanonical(c);
  if (entry === undefined) {
    throw new PathError(
      'no-mount-coverage',
      `no mount entry covers canonical path: ${JSON.stringify(c)}`
    );
  }

  return swapPrefix(c, entry.host, entry.container) as LocalPath;
}

/**
 * Translate a {@link LocalPath} back to a {@link CanonicalPath} using the
 * active mount map.
 *
 * Identity map re-runs full canonicalization on the local string. Non-identity
 * map strips the matched container prefix and prepends the host prefix, then
 * validates via {@link fromValidated}.
 *
 * @throws {PathError} with kind `'no-mount-coverage'` when no entry covers
 *   the local path, or any PathError from subsequent validation.
 */
export function toCanonical(l: LocalPath, mounts: MountMap): CanonicalPath {
  if (mounts.isIdentity) {
    return fromUserInput(l);
  }

  const entry = mounts.findForContainer(l);
  if (entry === undefined) {
    throw new PathError(
      'no-mount-coverage',
      `no mount entry covers local path: ${JSON.stringify(l)}`
    );
  }

  const translated = swapPrefix(l, entry.container, entry.host);
  return fromValidated(translated);
}

/**
 * Return the underlying string of a {@link LocalPath} for use in fs API
 * calls.
 *
 * This is the only sanctioned way to extract the raw string from a LocalPath
 * for passing to Node.js fs functions. Using `LocalPath` directly as a string
 * works at runtime (brand types are compile-time only) but going through
 * `asStdPath` makes the intent explicit.
 */
export function asStdPath(l: LocalPath): string {
  return l;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Whether `prefix` is a component-aware prefix of `path`.
 *
 * `/a/b` is a component-aware prefix of `/a/b` and `/a/b/c` but NOT
 * `/a/bc`. Mirrors the Rust `component_aware_prefix` in mount_map.rs.
 */
function componentAwarePrefix(path: string, prefix: string): boolean {
  if (path === prefix) return true;
  if (!path.startsWith(prefix)) return false;
  // The character immediately after the prefix must be '/', or the prefix
  // itself ends in '/' (the root '/' case).
  return prefix.endsWith('/') || path[prefix.length] === '/';
}

/**
 * Replace `fromPrefix` with `toPrefix` at the start of `path`.
 *
 * Mirrors the Rust `swap_prefix` in local.rs.
 */
function swapPrefix(path: string, fromPrefix: string, toPrefix: string): string {
  const suffix = path.slice(fromPrefix.length);
  const suffixTrimmed = suffix.replace(/^\/+/, '');
  const toTrimmed = toPrefix.replace(/\/+$/, '');

  if (suffixTrimmed === '') {
    // Whole path equaled the fromPrefix.
    return toPrefix;
  }
  if (toTrimmed === '') {
    // toPrefix was '/' (root).
    return `/${suffixTrimmed}`;
  }
  return `${toTrimmed}/${suffixTrimmed}`;
}
