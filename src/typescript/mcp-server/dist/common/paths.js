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
    _entries;
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
    constructor(rawEntries) {
        const entries = [];
        const seenHosts = new Set();
        const seenContainers = new Set();
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
    get isIdentity() {
        return this._entries.length === 0;
    }
    /** Number of declared mount entries. */
    get size() {
        return this._entries.length;
    }
    /**
     * Find the entry whose `host` prefix best covers the canonical path.
     * Longest-prefix-wins, component-aware (spec §5.2).
     */
    findForCanonical(canonical) {
        let best;
        for (const entry of this._entries) {
            if (componentAwarePrefix(canonical, entry.host) &&
                (best === undefined || entry.host.length > best.host.length)) {
                best = entry;
            }
        }
        return best;
    }
    /**
     * Find the entry whose `container` prefix best covers the local path.
     * Longest-prefix-wins, component-aware.
     */
    findForContainer(localStr) {
        let best;
        for (const entry of this._entries) {
            if (componentAwarePrefix(localStr, entry.container) &&
                (best === undefined || entry.container.length > best.container.length)) {
                best = entry;
            }
        }
        return best;
    }
}
/**
 * Failure modes for canonical-path construction and mount-map translation.
 *
 * Mirrors `wqm_common::paths::PathError` error variants (spec §4.1).
 */
export class PathError extends Error {
    kind;
    constructor(kind, message) {
        super(message);
        this.name = 'PathError';
        this.kind = kind;
    }
}
// ---------------------------------------------------------------------------
// §3.1 — Core normalization (private helper)
// ---------------------------------------------------------------------------
/**
 * Apply the nine normalization rules from spec §3.1 and return the canonical
 * string form.
 *
 * Pure string operation — no filesystem access, no symlink resolution.
 *
 * @throws {PathError} on any validation failure.
 */
function normalizePath(input) {
    // Rule 9 pre-check: embedded NUL bytes are never valid.
    if (input.includes('\0')) {
        throw new PathError('nul-byte', 'path contains embedded NUL byte');
    }
    // Empty input rejected before any further processing.
    if (input.length === 0) {
        throw new PathError('empty', 'path is empty');
    }
    // Rule 2: expand leading `~` to the user's home directory.
    let expanded;
    if (input === '~') {
        expanded = homedir();
    }
    else if (input.startsWith('~/')) {
        expanded = homedir() + input.slice(1);
    }
    else {
        expanded = input;
    }
    // Rule 1: must be absolute after tilde expansion.
    if (!expanded.startsWith('/')) {
        throw new PathError('relative', `path must be absolute, got: ${JSON.stringify(input)}`);
    }
    // Split on '/' and process each segment.
    // input = '/a//b/./c' → segments = ['', 'a', '', 'b', '.', 'c']
    const rawSegments = expanded.split('/');
    const parts = [];
    for (const seg of rawSegments) {
        if (seg === '' || seg === '.') {
            // Rules 3 & 5: drop empty segments (duplicate slashes) and '.' segments.
            continue;
        }
        // Rule 4: reject '..' entirely.
        if (seg === '..') {
            throw new PathError('dot-dot', `path must not contain '..' segments, got: ${JSON.stringify(input)}`);
        }
        // Rule 9: lone surrogates in JS strings cannot be represented as valid
        // UTF-8. Detect via encodeURIComponent which throws on lone surrogates.
        try {
            encodeURIComponent(seg);
        }
        catch {
            throw new PathError('non-utf8', 'path contains non-UTF-8 sequences (lone surrogate)');
        }
        // Rules 6 & 7: preserve case exactly; never resolve symlinks.
        parts.push(seg);
    }
    // Reconstruct with leading '/'.
    const result = '/' + parts.join('/');
    // An absolute path that reduces to just '/' is valid (root itself).
    return result;
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
export function fromUserInput(s) {
    return normalizePath(s);
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
export function fromValidated(s) {
    // Same checks as fromUserInput. For a value from a DB row the input should
    // already satisfy all rules; this call is the compile-time and runtime
    // safety net.
    return normalizePath(s);
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
export function toLocal(c, mounts) {
    if (mounts.isIdentity) {
        return c;
    }
    const entry = mounts.findForCanonical(c);
    if (entry === undefined) {
        throw new PathError('no-mount-coverage', `no mount entry covers canonical path: ${JSON.stringify(c)}`);
    }
    return swapPrefix(c, entry.host, entry.container);
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
export function toCanonical(l, mounts) {
    if (mounts.isIdentity) {
        return fromUserInput(l);
    }
    const entry = mounts.findForContainer(l);
    if (entry === undefined) {
        throw new PathError('no-mount-coverage', `no mount entry covers local path: ${JSON.stringify(l)}`);
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
export function asStdPath(l) {
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
function componentAwarePrefix(path, prefix) {
    if (path === prefix)
        return true;
    if (!path.startsWith(prefix))
        return false;
    // The character immediately after the prefix must be '/', or the prefix
    // itself ends in '/' (the root '/' case).
    return prefix.endsWith('/') || path[prefix.length] === '/';
}
/**
 * Replace `fromPrefix` with `toPrefix` at the start of `path`.
 *
 * Mirrors the Rust `swap_prefix` in local.rs.
 */
function swapPrefix(path, fromPrefix, toPrefix) {
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
//# sourceMappingURL=paths.js.map