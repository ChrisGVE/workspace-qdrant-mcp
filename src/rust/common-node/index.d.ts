/* TypeScript type definitions for wqm-common-node native addon */

// Project ID
export function calculateProjectId(projectRoot: string, gitRemote: string | null): string
export function calculateProjectIdWithDisambiguation(projectRoot: string, gitRemote: string | null, disambiguationPath: string | null): string
export function normalizeGitUrl(url: string): string
export function detectGitRemote(projectRoot: string): string | null
export function calculateTenantId(projectRoot: string): string

// Hashing
export function generateIdempotencyKey(itemType: string, op: string, tenantId: string, collection: string, payloadJson: string): string | null
export function computeContentHash(content: string): string

// NLP
export function tokenize(text: string): string[]

// Constants
export function collectionProjects(): string
export function collectionLibraries(): string
export function collectionMemory(): string
export function defaultQdrantUrl(): string
export function defaultGrpcPort(): number
export function defaultBranch(): string

// Queue type validation
export function isValidItemType(s: string): boolean
export function isValidQueueOperation(s: string): boolean
export function isValidQueueStatus(s: string): boolean
export function isValidOperationForType(itemType: string, op: string): boolean
