/** gRPC enum types — matches workspace_daemon.proto definitions. */
export declare enum ServiceStatus {
    SERVICE_STATUS_UNSPECIFIED = 0,
    SERVICE_STATUS_HEALTHY = 1,
    SERVICE_STATUS_DEGRADED = 2,
    SERVICE_STATUS_UNHEALTHY = 3,
    SERVICE_STATUS_UNAVAILABLE = 4
}
export declare enum QueueType {
    QUEUE_TYPE_UNSPECIFIED = 0,
    INGEST_QUEUE = 1,
    WATCHED_PROJECTS = 2,
    WATCHED_FOLDERS = 3,
    TOOLS_AVAILABLE = 4
}
export declare enum ServerState {
    SERVER_STATE_UNSPECIFIED = 0,
    SERVER_STATE_UP = 1,
    SERVER_STATE_DOWN = 2
}
//# sourceMappingURL=grpc-types-messages-enums.d.ts.map