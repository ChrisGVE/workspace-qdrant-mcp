/**
 * Store tool argument builder — parse raw MCP tool arguments into StoreOptions
 */
import type { SessionState } from '../server-types.js';
export type StoreOptions = {
    content: string;
    libraryName?: string;
    forProject?: boolean;
    projectId?: string;
    title?: string;
    url?: string;
    filePath?: string;
    sourceType?: 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';
    metadata?: Record<string, string>;
};
/**
 * Build store options from raw tool arguments.
 * Store tool is for libraries collection ONLY per spec.
 */
export declare function buildStoreOptions(args: Record<string, unknown> | undefined, sessionState: Pick<SessionState, 'projectId'>): StoreOptions;
//# sourceMappingURL=store.d.ts.map