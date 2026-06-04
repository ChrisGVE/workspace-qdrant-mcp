/** gRPC enum types — matches workspace_daemon.proto definitions. */
export var ServiceStatus;
(function (ServiceStatus) {
    ServiceStatus[ServiceStatus["SERVICE_STATUS_UNSPECIFIED"] = 0] = "SERVICE_STATUS_UNSPECIFIED";
    ServiceStatus[ServiceStatus["SERVICE_STATUS_HEALTHY"] = 1] = "SERVICE_STATUS_HEALTHY";
    ServiceStatus[ServiceStatus["SERVICE_STATUS_DEGRADED"] = 2] = "SERVICE_STATUS_DEGRADED";
    ServiceStatus[ServiceStatus["SERVICE_STATUS_UNHEALTHY"] = 3] = "SERVICE_STATUS_UNHEALTHY";
    ServiceStatus[ServiceStatus["SERVICE_STATUS_UNAVAILABLE"] = 4] = "SERVICE_STATUS_UNAVAILABLE";
})(ServiceStatus || (ServiceStatus = {}));
export var QueueType;
(function (QueueType) {
    QueueType[QueueType["QUEUE_TYPE_UNSPECIFIED"] = 0] = "QUEUE_TYPE_UNSPECIFIED";
    QueueType[QueueType["INGEST_QUEUE"] = 1] = "INGEST_QUEUE";
    QueueType[QueueType["WATCHED_PROJECTS"] = 2] = "WATCHED_PROJECTS";
    QueueType[QueueType["WATCHED_FOLDERS"] = 3] = "WATCHED_FOLDERS";
    QueueType[QueueType["TOOLS_AVAILABLE"] = 4] = "TOOLS_AVAILABLE";
})(QueueType || (QueueType = {}));
export var ServerState;
(function (ServerState) {
    ServerState[ServerState["SERVER_STATE_UNSPECIFIED"] = 0] = "SERVER_STATE_UNSPECIFIED";
    ServerState[ServerState["SERVER_STATE_UP"] = 1] = "SERVER_STATE_UP";
    ServerState[ServerState["SERVER_STATE_DOWN"] = 2] = "SERVER_STATE_DOWN";
})(ServerState || (ServerState = {}));
//# sourceMappingURL=grpc-types-messages-enums.js.map