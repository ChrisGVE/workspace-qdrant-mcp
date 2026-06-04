/**
 * List tool argument builder — parse raw MCP tool arguments into ListOptions
 */
import type { ListOptions } from '../tools/list-files-types.js';
export type { ListOptions };
/**
 * Build list options from raw tool arguments.
 *
 * @param args           Raw MCP tool arguments.
 * @param defaultBranch  Session's current branch, used when the caller does
 *                       not explicitly pass a `branch` argument. Pass `null`
 *                       or omit to skip the default. Pass the string `"*"` as
 *                       the `branch` argument to list files across all branches.
 */
export declare function buildListOptions(args: Record<string, unknown> | undefined, defaultBranch?: string | null): ListOptions;
//# sourceMappingURL=list.d.ts.map