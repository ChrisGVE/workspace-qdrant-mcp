import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

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

class ProjectPriority(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROJECT_PRIORITY_UNSPECIFIED: _ClassVar[ProjectPriority]
    PROJECT_PRIORITY_HIGH: _ClassVar[ProjectPriority]
    PROJECT_PRIORITY_NORMAL: _ClassVar[ProjectPriority]
    PROJECT_PRIORITY_LOW: _ClassVar[ProjectPriority]

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

class FileType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILE_TYPE_UNSPECIFIED: _ClassVar[FileType]
    FILE_TYPE_CODE: _ClassVar[FileType]
    FILE_TYPE_DOC: _ClassVar[FileType]
    FILE_TYPE_TEST: _ClassVar[FileType]
    FILE_TYPE_CONFIG: _ClassVar[FileType]
    FILE_TYPE_NOTE: _ClassVar[FileType]
    FILE_TYPE_ARTIFACT: _ClassVar[FileType]

class LibraryFileType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LIBRARY_FILE_TYPE_UNSPECIFIED: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_PDF: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_EPUB: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_MD: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_TXT: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_HTML: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_RST: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_DOC: _ClassVar[LibraryFileType]
    LIBRARY_FILE_TYPE_DOCX: _ClassVar[LibraryFileType]

class ContentSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTENT_SOURCE_UNSPECIFIED: _ClassVar[ContentSource]
    CONTENT_SOURCE_FILE: _ClassVar[ContentSource]
    CONTENT_SOURCE_USER_INPUT: _ClassVar[ContentSource]
    CONTENT_SOURCE_WEB: _ClassVar[ContentSource]
    CONTENT_SOURCE_CHAT: _ClassVar[ContentSource]
    CONTENT_SOURCE_GENERATED: _ClassVar[ContentSource]

class RuleType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RULE_TYPE_UNSPECIFIED: _ClassVar[RuleType]
    RULE_TYPE_PREFERENCE: _ClassVar[RuleType]
    RULE_TYPE_BEHAVIOR: _ClassVar[RuleType]
    RULE_TYPE_CONSTRAINT: _ClassVar[RuleType]
    RULE_TYPE_PATTERN: _ClassVar[RuleType]

class RuleScope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RULE_SCOPE_UNSPECIFIED: _ClassVar[RuleScope]
    RULE_SCOPE_GLOBAL: _ClassVar[RuleScope]
    RULE_SCOPE_PROJECT: _ClassVar[RuleScope]
    RULE_SCOPE_LANGUAGE: _ClassVar[RuleScope]

class PriorityLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PRIORITY_LEVEL_UNSPECIFIED: _ClassVar[PriorityLevel]
    PRIORITY_LEVEL_LOW: _ClassVar[PriorityLevel]
    PRIORITY_LEVEL_NORMAL: _ClassVar[PriorityLevel]
    PRIORITY_LEVEL_HIGH: _ClassVar[PriorityLevel]
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
PROJECT_PRIORITY_UNSPECIFIED: ProjectPriority
PROJECT_PRIORITY_HIGH: ProjectPriority
PROJECT_PRIORITY_NORMAL: ProjectPriority
PROJECT_PRIORITY_LOW: ProjectPriority
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
FILE_TYPE_UNSPECIFIED: FileType
FILE_TYPE_CODE: FileType
FILE_TYPE_DOC: FileType
FILE_TYPE_TEST: FileType
FILE_TYPE_CONFIG: FileType
FILE_TYPE_NOTE: FileType
FILE_TYPE_ARTIFACT: FileType
LIBRARY_FILE_TYPE_UNSPECIFIED: LibraryFileType
LIBRARY_FILE_TYPE_PDF: LibraryFileType
LIBRARY_FILE_TYPE_EPUB: LibraryFileType
LIBRARY_FILE_TYPE_MD: LibraryFileType
LIBRARY_FILE_TYPE_TXT: LibraryFileType
LIBRARY_FILE_TYPE_HTML: LibraryFileType
LIBRARY_FILE_TYPE_RST: LibraryFileType
LIBRARY_FILE_TYPE_DOC: LibraryFileType
LIBRARY_FILE_TYPE_DOCX: LibraryFileType
CONTENT_SOURCE_UNSPECIFIED: ContentSource
CONTENT_SOURCE_FILE: ContentSource
CONTENT_SOURCE_USER_INPUT: ContentSource
CONTENT_SOURCE_WEB: ContentSource
CONTENT_SOURCE_CHAT: ContentSource
CONTENT_SOURCE_GENERATED: ContentSource
RULE_TYPE_UNSPECIFIED: RuleType
RULE_TYPE_PREFERENCE: RuleType
RULE_TYPE_BEHAVIOR: RuleType
RULE_TYPE_CONSTRAINT: RuleType
RULE_TYPE_PATTERN: RuleType
RULE_SCOPE_UNSPECIFIED: RuleScope
RULE_SCOPE_GLOBAL: RuleScope
RULE_SCOPE_PROJECT: RuleScope
RULE_SCOPE_LANGUAGE: RuleScope
PRIORITY_LEVEL_UNSPECIFIED: PriorityLevel
PRIORITY_LEVEL_LOW: PriorityLevel
PRIORITY_LEVEL_NORMAL: PriorityLevel
PRIORITY_LEVEL_HIGH: PriorityLevel

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "components", "timestamp")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    status: ServiceStatus
    components: _containers.RepeatedCompositeFieldContainer[ComponentHealth]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, status: _Optional[_Union[ServiceStatus, str]] = ..., components: _Optional[_Iterable[_Union[ComponentHealth, _Mapping]]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

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
    def __init__(self, component_name: _Optional[str] = ..., status: _Optional[_Union[ServiceStatus, str]] = ..., message: _Optional[str] = ..., last_check: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

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
    def __init__(self, status: _Optional[_Union[ServiceStatus, str]] = ..., metrics: _Optional[_Union[SystemMetrics, _Mapping]] = ..., active_projects: _Optional[_Iterable[str]] = ..., total_documents: _Optional[int] = ..., total_collections: _Optional[int] = ..., uptime_since: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

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
    def __init__(self, cpu_usage_percent: _Optional[float] = ..., memory_usage_bytes: _Optional[int] = ..., memory_total_bytes: _Optional[int] = ..., disk_usage_bytes: _Optional[int] = ..., disk_total_bytes: _Optional[int] = ..., active_connections: _Optional[int] = ..., pending_operations: _Optional[int] = ...) -> None: ...

class MetricsResponse(_message.Message):
    __slots__ = ("metrics", "collected_at")
    METRICS_FIELD_NUMBER: _ClassVar[int]
    COLLECTED_AT_FIELD_NUMBER: _ClassVar[int]
    metrics: _containers.RepeatedCompositeFieldContainer[Metric]
    collected_at: _timestamp_pb2.Timestamp
    def __init__(self, metrics: _Optional[_Iterable[_Union[Metric, _Mapping]]] = ..., collected_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Metric(_message.Message):
    __slots__ = ("name", "type", "labels", "value", "timestamp")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
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
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ..., value: _Optional[float] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class RefreshSignalRequest(_message.Message):
    __slots__ = ("queue_type", "lsp_languages", "grammar_languages")
    QUEUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    LSP_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    GRAMMAR_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    queue_type: QueueType
    lsp_languages: _containers.RepeatedScalarFieldContainer[str]
    grammar_languages: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, queue_type: _Optional[_Union[QueueType, str]] = ..., lsp_languages: _Optional[_Iterable[str]] = ..., grammar_languages: _Optional[_Iterable[str]] = ...) -> None: ...

class ServerStatusNotification(_message.Message):
    __slots__ = ("state", "project_name", "project_root")
    STATE_FIELD_NUMBER: _ClassVar[int]
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    state: ServerState
    project_name: str
    project_root: str
    def __init__(self, state: _Optional[_Union[ServerState, str]] = ..., project_name: _Optional[str] = ..., project_root: _Optional[str] = ...) -> None: ...

class CreateCollectionRequest(_message.Message):
    __slots__ = ("collection_name", "project_id", "config")
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    project_id: str
    config: CollectionConfig
    def __init__(self, collection_name: _Optional[str] = ..., project_id: _Optional[str] = ..., config: _Optional[_Union[CollectionConfig, _Mapping]] = ...) -> None: ...

class CreateCollectionResponse(_message.Message):
    __slots__ = ("success", "error_message", "collection_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    collection_id: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ..., collection_id: _Optional[str] = ...) -> None: ...

class CollectionConfig(_message.Message):
    __slots__ = ("vector_size", "distance_metric", "enable_indexing", "metadata_schema")
    class MetadataSchemaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VECTOR_SIZE_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    ENABLE_INDEXING_FIELD_NUMBER: _ClassVar[int]
    METADATA_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    vector_size: int
    distance_metric: str
    enable_indexing: bool
    metadata_schema: _containers.ScalarMap[str, str]
    def __init__(self, vector_size: _Optional[int] = ..., distance_metric: _Optional[str] = ..., enable_indexing: bool = ..., metadata_schema: _Optional[_Mapping[str, str]] = ...) -> None: ...

class DeleteCollectionRequest(_message.Message):
    __slots__ = ("collection_name", "project_id", "force")
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    project_id: str
    force: bool
    def __init__(self, collection_name: _Optional[str] = ..., project_id: _Optional[str] = ..., force: bool = ...) -> None: ...

class CreateAliasRequest(_message.Message):
    __slots__ = ("alias_name", "collection_name")
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    collection_name: str
    def __init__(self, alias_name: _Optional[str] = ..., collection_name: _Optional[str] = ...) -> None: ...

class DeleteAliasRequest(_message.Message):
    __slots__ = ("alias_name",)
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    def __init__(self, alias_name: _Optional[str] = ...) -> None: ...

class RenameAliasRequest(_message.Message):
    __slots__ = ("old_alias_name", "new_alias_name", "collection_name")
    OLD_ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    old_alias_name: str
    new_alias_name: str
    collection_name: str
    def __init__(self, old_alias_name: _Optional[str] = ..., new_alias_name: _Optional[str] = ..., collection_name: _Optional[str] = ...) -> None: ...

class IngestTextRequest(_message.Message):
    __slots__ = ("content", "collection_basename", "tenant_id", "document_id", "metadata", "chunk_text")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
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
    def __init__(self, content: _Optional[str] = ..., collection_basename: _Optional[str] = ..., tenant_id: _Optional[str] = ..., document_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., chunk_text: bool = ...) -> None: ...

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
    def __init__(self, document_id: _Optional[str] = ..., success: bool = ..., chunks_created: _Optional[int] = ..., error_message: _Optional[str] = ...) -> None: ...

class UpdateTextRequest(_message.Message):
    __slots__ = ("document_id", "content", "collection_name", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    content: str
    collection_name: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, document_id: _Optional[str] = ..., content: _Optional[str] = ..., collection_name: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class UpdateTextResponse(_message.Message):
    __slots__ = ("success", "error_message", "updated_at")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class DeleteTextRequest(_message.Message):
    __slots__ = ("document_id", "collection_name")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    collection_name: str
    def __init__(self, document_id: _Optional[str] = ..., collection_name: _Optional[str] = ...) -> None: ...

class RegisterProjectRequest(_message.Message):
    __slots__ = ("path", "project_id", "name", "git_remote")
    PATH_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    GIT_REMOTE_FIELD_NUMBER: _ClassVar[int]
    path: str
    project_id: str
    name: str
    git_remote: str
    def __init__(self, path: _Optional[str] = ..., project_id: _Optional[str] = ..., name: _Optional[str] = ..., git_remote: _Optional[str] = ...) -> None: ...

class RegisterProjectResponse(_message.Message):
    __slots__ = ("created", "project_id", "priority", "active_sessions")
    CREATED_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    created: bool
    project_id: str
    priority: str
    active_sessions: int
    def __init__(self, created: bool = ..., project_id: _Optional[str] = ..., priority: _Optional[str] = ..., active_sessions: _Optional[int] = ...) -> None: ...

class DeprioritizeProjectRequest(_message.Message):
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: _Optional[str] = ...) -> None: ...

class DeprioritizeProjectResponse(_message.Message):
    __slots__ = ("success", "remaining_sessions", "new_priority")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    REMAINING_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    NEW_PRIORITY_FIELD_NUMBER: _ClassVar[int]
    success: bool
    remaining_sessions: int
    new_priority: str
    def __init__(self, success: bool = ..., remaining_sessions: _Optional[int] = ..., new_priority: _Optional[str] = ...) -> None: ...

class GetProjectStatusRequest(_message.Message):
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: _Optional[str] = ...) -> None: ...

class GetProjectStatusResponse(_message.Message):
    __slots__ = ("found", "project_id", "project_name", "project_root", "priority", "active_sessions", "last_active", "registered_at", "git_remote")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_AT_FIELD_NUMBER: _ClassVar[int]
    GIT_REMOTE_FIELD_NUMBER: _ClassVar[int]
    found: bool
    project_id: str
    project_name: str
    project_root: str
    priority: str
    active_sessions: int
    last_active: _timestamp_pb2.Timestamp
    registered_at: _timestamp_pb2.Timestamp
    git_remote: str
    def __init__(self, found: bool = ..., project_id: _Optional[str] = ..., project_name: _Optional[str] = ..., project_root: _Optional[str] = ..., priority: _Optional[str] = ..., active_sessions: _Optional[int] = ..., last_active: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., registered_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., git_remote: _Optional[str] = ...) -> None: ...

class ListProjectsRequest(_message.Message):
    __slots__ = ("priority_filter", "active_only")
    PRIORITY_FILTER_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_ONLY_FIELD_NUMBER: _ClassVar[int]
    priority_filter: str
    active_only: bool
    def __init__(self, priority_filter: _Optional[str] = ..., active_only: bool = ...) -> None: ...

class ListProjectsResponse(_message.Message):
    __slots__ = ("projects", "total_count")
    PROJECTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    projects: _containers.RepeatedCompositeFieldContainer[ProjectInfo]
    total_count: int
    def __init__(self, projects: _Optional[_Iterable[_Union[ProjectInfo, _Mapping]]] = ..., total_count: _Optional[int] = ...) -> None: ...

class ProjectInfo(_message.Message):
    __slots__ = ("project_id", "project_name", "project_root", "priority", "active_sessions", "last_active")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    project_name: str
    project_root: str
    priority: str
    active_sessions: int
    last_active: _timestamp_pb2.Timestamp
    def __init__(self, project_id: _Optional[str] = ..., project_name: _Optional[str] = ..., project_root: _Optional[str] = ..., priority: _Optional[str] = ..., active_sessions: _Optional[int] = ..., last_active: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("acknowledged", "next_heartbeat_by")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    NEXT_HEARTBEAT_BY_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    next_heartbeat_by: _timestamp_pb2.Timestamp
    def __init__(self, acknowledged: bool = ..., next_heartbeat_by: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ProjectPayload(_message.Message):
    __slots__ = ("project_id", "project_name", "file_path", "file_absolute_path", "file_type", "language", "branch", "symbols", "symbols_defined", "symbols_used", "title", "source", "lsp_metadata", "chunk_index", "total_chunks", "created_at", "updated_at")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    FILE_ABSOLUTE_PATH_FIELD_NUMBER: _ClassVar[int]
    FILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    BRANCH_FIELD_NUMBER: _ClassVar[int]
    SYMBOLS_FIELD_NUMBER: _ClassVar[int]
    SYMBOLS_DEFINED_FIELD_NUMBER: _ClassVar[int]
    SYMBOLS_USED_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    LSP_METADATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    project_name: str
    file_path: str
    file_absolute_path: str
    file_type: FileType
    language: str
    branch: str
    symbols: _containers.RepeatedScalarFieldContainer[str]
    symbols_defined: _containers.RepeatedScalarFieldContainer[str]
    symbols_used: _containers.RepeatedScalarFieldContainer[str]
    title: str
    source: ContentSource
    lsp_metadata: LspMetadata
    chunk_index: int
    total_chunks: int
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, project_id: _Optional[str] = ..., project_name: _Optional[str] = ..., file_path: _Optional[str] = ..., file_absolute_path: _Optional[str] = ..., file_type: _Optional[_Union[FileType, str]] = ..., language: _Optional[str] = ..., branch: _Optional[str] = ..., symbols: _Optional[_Iterable[str]] = ..., symbols_defined: _Optional[_Iterable[str]] = ..., symbols_used: _Optional[_Iterable[str]] = ..., title: _Optional[str] = ..., source: _Optional[_Union[ContentSource, str]] = ..., lsp_metadata: _Optional[_Union[LspMetadata, _Mapping]] = ..., chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class LspMetadata(_message.Message):
    __slots__ = ("definitions", "references", "hover_info")
    DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    HOVER_INFO_FIELD_NUMBER: _ClassVar[int]
    definitions: _containers.RepeatedCompositeFieldContainer[SymbolDefinition]
    references: _containers.RepeatedCompositeFieldContainer[SymbolReference]
    hover_info: str
    def __init__(self, definitions: _Optional[_Iterable[_Union[SymbolDefinition, _Mapping]]] = ..., references: _Optional[_Iterable[_Union[SymbolReference, _Mapping]]] = ..., hover_info: _Optional[str] = ...) -> None: ...

class SymbolDefinition(_message.Message):
    __slots__ = ("name", "kind", "line", "column")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LINE_FIELD_NUMBER: _ClassVar[int]
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: str
    line: int
    column: int
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[str] = ..., line: _Optional[int] = ..., column: _Optional[int] = ...) -> None: ...

class SymbolReference(_message.Message):
    __slots__ = ("name", "file_path", "line")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    LINE_FIELD_NUMBER: _ClassVar[int]
    name: str
    file_path: str
    line: int
    def __init__(self, name: _Optional[str] = ..., file_path: _Optional[str] = ..., line: _Optional[int] = ...) -> None: ...

class LibraryPayload(_message.Message):
    __slots__ = ("library_name", "source_file", "file_type", "title", "author", "topics", "folder", "library_version", "page_number", "chunk_index", "total_chunks", "created_at")
    LIBRARY_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FILE_FIELD_NUMBER: _ClassVar[int]
    FILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    FOLDER_FIELD_NUMBER: _ClassVar[int]
    LIBRARY_VERSION_FIELD_NUMBER: _ClassVar[int]
    PAGE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    library_name: str
    source_file: str
    file_type: LibraryFileType
    title: str
    author: str
    topics: _containers.RepeatedScalarFieldContainer[str]
    folder: str
    library_version: str
    page_number: int
    chunk_index: int
    total_chunks: int
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, library_name: _Optional[str] = ..., source_file: _Optional[str] = ..., file_type: _Optional[_Union[LibraryFileType, str]] = ..., title: _Optional[str] = ..., author: _Optional[str] = ..., topics: _Optional[_Iterable[str]] = ..., folder: _Optional[str] = ..., library_version: _Optional[str] = ..., page_number: _Optional[int] = ..., chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class MemoryPayload(_message.Message):
    __slots__ = ("rule_id", "rule_type", "content", "priority", "scope", "project_id", "language", "enabled", "created_at", "updated_at")
    RULE_ID_FIELD_NUMBER: _ClassVar[int]
    RULE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    rule_id: str
    rule_type: RuleType
    content: str
    priority: int
    scope: RuleScope
    project_id: str
    language: str
    enabled: bool
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, rule_id: _Optional[str] = ..., rule_type: _Optional[_Union[RuleType, str]] = ..., content: _Optional[str] = ..., priority: _Optional[int] = ..., scope: _Optional[_Union[RuleScope, str]] = ..., project_id: _Optional[str] = ..., language: _Optional[str] = ..., enabled: bool = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
