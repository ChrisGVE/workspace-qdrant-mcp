/**
 * gRPC client for communicating with the Rust daemon (memexd).
 *
 * Provides type-safe wrappers around daemon RPC methods with automatic
 * retry (exponential backoff), connection health monitoring, and timeouts.
 *
 * Implementation is split across:
 *   daemon-client/connection.ts     — base class, lifecycle, retry logic
 *   daemon-client/system-methods.ts — SystemService + ProjectService RPCs
 *   daemon-client/service-methods.ts — remaining service RPCs
 */
export { DaemonClientService as DaemonClient } from './daemon-client/service-methods.js';
// Re-export types for convenience
export { ServiceStatus } from './grpc-types.js';
//# sourceMappingURL=daemon-client.js.map