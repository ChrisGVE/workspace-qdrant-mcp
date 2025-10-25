import datetime
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ServiceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVICE_STATUS_UNSPECIFIED: _ClassVar[ServiceStatus]
    SERVICE_STATUS_HEALTHY: _ClassVar[ServiceStatus]
    SERVICE_STATUS_DEGRADED: _ClassVar[ServiceStatus]
    SERVICE_STATUS_UNHEALTHY: _ClassVar[ServiceStatus]
    SERVICE_STATUS_UNAVAILABLE: _ClassVar[ServiceStatus]

class QueueType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    QUEUE_TYPE_UNSPECIFIED: _ClassVar[QueueType]
    INGEST_QUEUE: _ClassVar[QueueType]
    WATCHED_PROJECTS: _ClassVar[QueueType]
    WATCHED_FOLDERS: _ClassVar[QueueType]
    TOOLS_AVAILABLE: _ClassVar[QueueType]

class ServerState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVER_STATE_UNSPECIFIED: _ClassVar[ServerState]
    SERVER_STATE_UP: _ClassVar[ServerState]
    SERVER_STATE_DOWN: _ClassVar[ServerState]

class DocumentType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DOCUMENT_TYPE_UNSPECIFIED: _ClassVar[DocumentType]
    DOCUMENT_TYPE_CODE: _ClassVar[DocumentType]
    DOCUMENT_TYPE_PDF: _ClassVar[DocumentType]
    DOCUMENT_TYPE_EPUB: _ClassVar[DocumentType]
    DOCUMENT_TYPE_MOBI: _ClassVar[DocumentType]
    DOCUMENT_TYPE_HTML: _ClassVar[DocumentType]
    DOCUMENT_TYPE_TEXT: _ClassVar[DocumentType]
    DOCUMENT_TYPE_MARKDOWN: _ClassVar[DocumentType]
    DOCUMENT_TYPE_JSON: _ClassVar[DocumentType]
    DOCUMENT_TYPE_XML: _ClassVar[DocumentType]

class ProcessingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROCESSING_STATUS_UNSPECIFIED: _ClassVar[ProcessingStatus]
    PROCESSING_STATUS_PENDING: _ClassVar[ProcessingStatus]
    PROCESSING_STATUS_IN_PROGRESS: _ClassVar[ProcessingStatus]
    PROCESSING_STATUS_COMPLETED: _ClassVar[ProcessingStatus]
    PROCESSING_STATUS_FAILED: _ClassVar[ProcessingStatus]
    PROCESSING_STATUS_CANCELLED: _ClassVar[ProcessingStatus]
SERVICE_STATUS_UNSPECIFIED: ServiceStatus
SERVICE_STATUS_HEALTHY: ServiceStatus
SERVICE_STATUS_DEGRADED: ServiceStatus
SERVICE_STATUS_UNHEALTHY: ServiceStatus
SERVICE_STATUS_UNAVAILABLE: ServiceStatus
QUEUE_TYPE_UNSPECIFIED: QueueType
INGEST_QUEUE: QueueType
WATCHED_PROJECTS: QueueType
WATCHED_FOLDERS: QueueType
TOOLS_AVAILABLE: QueueType
SERVER_STATE_UNSPECIFIED: ServerState
SERVER_STATE_UP: ServerState
SERVER_STATE_DOWN: ServerState
DOCUMENT_TYPE_UNSPECIFIED: DocumentType
DOCUMENT_TYPE_CODE: DocumentType
DOCUMENT_TYPE_PDF: DocumentType
DOCUMENT_TYPE_EPUB: DocumentType
DOCUMENT_TYPE_MOBI: DocumentType
DOCUMENT_TYPE_HTML: DocumentType
DOCUMENT_TYPE_TEXT: DocumentType
DOCUMENT_TYPE_MARKDOWN: DocumentType
DOCUMENT_TYPE_JSON: DocumentType
DOCUMENT_TYPE_XML: DocumentType
PROCESSING_STATUS_UNSPECIFIED: ProcessingStatus
PROCESSING_STATUS_PENDING: ProcessingStatus
PROCESSING_STATUS_IN_PROGRESS: ProcessingStatus
PROCESSING_STATUS_COMPLETED: ProcessingStatus
PROCESSING_STATUS_FAILED: ProcessingStatus
PROCESSING_STATUS_CANCELLED: ProcessingStatus

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "components", "timestamp")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    status: ServiceStatus
    components: _containers.RepeatedCompositeFieldContainer[ComponentHealth]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, status: ServiceStatus | str | None = ..., components: _Iterable[ComponentHealth | _Mapping] | None = ..., timestamp: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class ComponentHealth(_message.Message):
    __slots__ = ("component_name", "status", "message", "last_check")
    COMPONENT_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LAST_CHECK_FIELD_NUMBER: _ClassVar[int]
    component_name: str
    status: ServiceStatus
    message: str
    last_check: _timestamp_pb2.Timestamp
    def __init__(self, component_name: str | None = ..., status: ServiceStatus | str | None = ..., message: str | None = ..., last_check: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class SystemStatusResponse(_message.Message):
    __slots__ = ("status", "metrics", "active_projects", "total_documents", "total_collections", "uptime_since")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_PROJECTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SINCE_FIELD_NUMBER: _ClassVar[int]
    status: ServiceStatus
    metrics: SystemMetrics
    active_projects: _containers.RepeatedScalarFieldContainer[str]
    total_documents: int
    total_collections: int
    uptime_since: _timestamp_pb2.Timestamp
    def __init__(self, status: ServiceStatus | str | None = ..., metrics: SystemMetrics | _Mapping | None = ..., active_projects: _Iterable[str] | None = ..., total_documents: int | None = ..., total_collections: int | None = ..., uptime_since: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class SystemMetrics(_message.Message):
    __slots__ = ("cpu_usage_percent", "memory_usage_bytes", "memory_total_bytes", "disk_usage_bytes", "disk_total_bytes", "active_connections", "pending_operations")
    CPU_USAGE_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_USAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_CONNECTIONS_FIELD_NUMBER: _ClassVar[int]
    PENDING_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    cpu_usage_percent: float
    memory_usage_bytes: int
    memory_total_bytes: int
    disk_usage_bytes: int
    disk_total_bytes: int
    active_connections: int
    pending_operations: int
    def __init__(self, cpu_usage_percent: float | None = ..., memory_usage_bytes: int | None = ..., memory_total_bytes: int | None = ..., disk_usage_bytes: int | None = ..., disk_total_bytes: int | None = ..., active_connections: int | None = ..., pending_operations: int | None = ...) -> None: ...

class MetricsResponse(_message.Message):
    __slots__ = ("metrics", "collected_at")
    METRICS_FIELD_NUMBER: _ClassVar[int]
    COLLECTED_AT_FIELD_NUMBER: _ClassVar[int]
    metrics: _containers.RepeatedCompositeFieldContainer[Metric]
    collected_at: _timestamp_pb2.Timestamp
    def __init__(self, metrics: _Iterable[Metric | _Mapping] | None = ..., collected_at: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class Metric(_message.Message):
    __slots__ = ("name", "type", "labels", "value", "timestamp")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    labels: _containers.ScalarMap[str, str]
    value: float
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, name: str | None = ..., type: str | None = ..., labels: _Mapping[str, str] | None = ..., value: float | None = ..., timestamp: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class RefreshSignalRequest(_message.Message):
    __slots__ = ("queue_type", "lsp_languages", "grammar_languages")
    QUEUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    LSP_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    GRAMMAR_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    queue_type: QueueType
    lsp_languages: _containers.RepeatedScalarFieldContainer[str]
    grammar_languages: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, queue_type: QueueType | str | None = ..., lsp_languages: _Iterable[str] | None = ..., grammar_languages: _Iterable[str] | None = ...) -> None: ...

class ServerStatusNotification(_message.Message):
    __slots__ = ("state", "project_name", "project_root")
    STATE_FIELD_NUMBER: _ClassVar[int]
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    state: ServerState
    project_name: str
    project_root: str
    def __init__(self, state: ServerState | str | None = ..., project_name: str | None = ..., project_root: str | None = ...) -> None: ...

class CreateCollectionRequest(_message.Message):
    __slots__ = ("collection_name", "project_id", "config")
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    project_id: str
    config: CollectionConfig
    def __init__(self, collection_name: str | None = ..., project_id: str | None = ..., config: CollectionConfig | _Mapping | None = ...) -> None: ...

class CreateCollectionResponse(_message.Message):
    __slots__ = ("success", "error_message", "collection_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    collection_id: str
    def __init__(self, success: bool = ..., error_message: str | None = ..., collection_id: str | None = ...) -> None: ...

class CollectionConfig(_message.Message):
    __slots__ = ("vector_size", "distance_metric", "enable_indexing", "metadata_schema")
    class MetadataSchemaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    VECTOR_SIZE_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    ENABLE_INDEXING_FIELD_NUMBER: _ClassVar[int]
    METADATA_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    vector_size: int
    distance_metric: str
    enable_indexing: bool
    metadata_schema: _containers.ScalarMap[str, str]
    def __init__(self, vector_size: int | None = ..., distance_metric: str | None = ..., enable_indexing: bool = ..., metadata_schema: _Mapping[str, str] | None = ...) -> None: ...

class DeleteCollectionRequest(_message.Message):
    __slots__ = ("collection_name", "project_id", "force")
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    project_id: str
    force: bool
    def __init__(self, collection_name: str | None = ..., project_id: str | None = ..., force: bool = ...) -> None: ...

class CreateAliasRequest(_message.Message):
    __slots__ = ("alias_name", "collection_name")
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    collection_name: str
    def __init__(self, alias_name: str | None = ..., collection_name: str | None = ...) -> None: ...

class DeleteAliasRequest(_message.Message):
    __slots__ = ("alias_name",)
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    def __init__(self, alias_name: str | None = ...) -> None: ...

class RenameAliasRequest(_message.Message):
    __slots__ = ("old_alias_name", "new_alias_name", "collection_name")
    OLD_ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    old_alias_name: str
    new_alias_name: str
    collection_name: str
    def __init__(self, old_alias_name: str | None = ..., new_alias_name: str | None = ..., collection_name: str | None = ...) -> None: ...

class IngestTextRequest(_message.Message):
    __slots__ = ("content", "collection_basename", "tenant_id", "document_id", "metadata", "chunk_text")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_BASENAME_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_TEXT_FIELD_NUMBER: _ClassVar[int]
    content: str
    collection_basename: str
    tenant_id: str
    document_id: str
    metadata: _containers.ScalarMap[str, str]
    chunk_text: bool
    def __init__(self, content: str | None = ..., collection_basename: str | None = ..., tenant_id: str | None = ..., document_id: str | None = ..., metadata: _Mapping[str, str] | None = ..., chunk_text: bool = ...) -> None: ...

class IngestTextResponse(_message.Message):
    __slots__ = ("document_id", "success", "chunks_created", "error_message")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_CREATED_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    success: bool
    chunks_created: int
    error_message: str
    def __init__(self, document_id: str | None = ..., success: bool = ..., chunks_created: int | None = ..., error_message: str | None = ...) -> None: ...

class UpdateTextRequest(_message.Message):
    __slots__ = ("document_id", "content", "collection_name", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    content: str
    collection_name: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, document_id: str | None = ..., content: str | None = ..., collection_name: str | None = ..., metadata: _Mapping[str, str] | None = ...) -> None: ...

class UpdateTextResponse(_message.Message):
    __slots__ = ("success", "error_message", "updated_at")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, success: bool = ..., error_message: str | None = ..., updated_at: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class DeleteTextRequest(_message.Message):
    __slots__ = ("document_id", "collection_name")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    collection_name: str
    def __init__(self, document_id: str | None = ..., collection_name: str | None = ...) -> None: ...
