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
export type CanonicalPath = string & {
    readonly [_canonical]: true;
};
/**
 * Path as seen by the current process's filesystem. Differs between host
 * and container deployments. Used only for fs I/O calls. Never serialized.
 *
 * Construct only via {@link toLocal}. Never cast with `as LocalPath`.
 */
export type LocalPath = string & {
    readonly [_local]: true;
};
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
export declare class MountMap {
    private readonly _entries;
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
    constructor(rawEntries: ReadonlyArray<{
        host: string;
        container: string;
    }>);
    /** True when the map has zero entries (identity / host-native mode). */
    get isIdentity(): boolean;
    /** Number of declared mount entries. */
    get size(): number;
    /**
     * Find the entry whose `host` prefix best covers the canonical path.
     * Longest-prefix-wins, component-aware (spec §5.2).
     */
    findForCanonical(canonical: string): MountEntry | undefined;
    /**
     * Find the entry whose `container` prefix best covers the local path.
     * Longest-prefix-wins, component-aware.
     */
    findForContainer(localStr: string): MountEntry | undefined;
}
/** Discriminant for the specific failure mode in {@link PathError}. */
export type PathErrorKind = 'relative' | 'dot-dot' | 'non-utf8' | 'empty' | 'nul-byte' | 'no-mount-coverage' | 'mount-duplicate' | 'invalid';
/**
 * Failure modes for canonical-path construction and mount-map translation.
 *
 * Mirrors `wqm_common::paths::PathError` error variants (spec §4.1).
 */
export declare class PathError extends Error {
    readonly kind: PathErrorKind;
    constructor(kind: PathErrorKind, message: string);
}
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
export declare function fromUserInput(s: string): CanonicalPath;
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
export declare function fromValidated(s: string): CanonicalPath;
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
export declare function toLocal(c: CanonicalPath, mounts: MountMap): LocalPath;
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
export declare function toCanonical(l: LocalPath, mounts: MountMap): CanonicalPath;
/**
 * Return the underlying string of a {@link LocalPath} for use in fs API
 * calls.
 *
 * This is the only sanctioned way to extract the raw string from a LocalPath
 * for passing to Node.js fs functions. Using `LocalPath` directly as a string
 * works at runtime (brand types are compile-time only) but going through
 * `asStdPath` makes the intent explicit.
 */
export declare function asStdPath(l: LocalPath): string;
export {};
//# sourceMappingURL=paths.d.ts.map