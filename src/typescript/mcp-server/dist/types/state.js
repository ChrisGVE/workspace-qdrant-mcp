/**
 * SQLite state database types
 * Matches the schema owned by the Rust daemon (ADR-003)
 */
export function createDegradedResponse(message) {
    return {
        results: [],
        status: 'degraded',
        reason: 'database_not_initialized',
        message: message ?? 'Daemon has not run yet. Results may be incomplete.',
    };
}
//# sourceMappingURL=state.js.map