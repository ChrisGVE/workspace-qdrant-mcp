//! Schema definition for the `unified_queue` SQLite table.

use crate::schema::{FieldDef, TableDef};

pub const TABLE: TableDef = TableDef { name: "unified_queue" };

pub const QUEUE_ID: FieldDef = FieldDef::categorical("queue_id");
pub const ITEM_TYPE: FieldDef = FieldDef::categorical("item_type");
pub const OP: FieldDef = FieldDef::categorical("op");
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");
pub const COLLECTION: FieldDef = FieldDef::categorical("collection");
pub const PRIORITY: FieldDef = FieldDef::categorical("priority");
pub const STATUS: FieldDef = FieldDef::categorical("status");
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");
pub const UPDATED_AT: FieldDef = FieldDef::categorical("updated_at");
pub const LEASE_UNTIL: FieldDef = FieldDef::categorical("lease_until");
pub const WORKER_ID: FieldDef = FieldDef::categorical("worker_id");
pub const IDEMPOTENCY_KEY: FieldDef = FieldDef::categorical("idempotency_key");
pub const PAYLOAD_JSON: FieldDef = FieldDef::content("payload_json");
pub const RETRY_COUNT: FieldDef = FieldDef::categorical("retry_count");
pub const MAX_RETRIES: FieldDef = FieldDef::categorical("max_retries");
pub const ERROR_MESSAGE: FieldDef = FieldDef::content("error_message");
pub const LAST_ERROR_AT: FieldDef = FieldDef::categorical("last_error_at");
pub const BRANCH: FieldDef = FieldDef::categorical("branch");
pub const METADATA: FieldDef = FieldDef::content("metadata");
pub const FILE_PATH: FieldDef = FieldDef::content("file_path");

/// All columns in definition order.
pub const ALL_COLUMNS: &[FieldDef] = &[
    QUEUE_ID, ITEM_TYPE, OP, TENANT_ID, COLLECTION,
    PRIORITY, STATUS, CREATED_AT, UPDATED_AT,
    LEASE_UNTIL, WORKER_ID, IDEMPOTENCY_KEY,
    PAYLOAD_JSON, RETRY_COUNT, MAX_RETRIES,
    ERROR_MESSAGE, LAST_ERROR_AT, BRANCH, METADATA, FILE_PATH,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FieldKind;

    #[test]
    fn table_name() {
        assert_eq!(TABLE.name, "unified_queue");
    }

    #[test]
    fn column_count() {
        assert_eq!(ALL_COLUMNS.len(), 20);
    }

    #[test]
    fn content_columns_are_content() {
        let content: Vec<&str> = ALL_COLUMNS.iter()
            .filter(|f| matches!(f.kind, FieldKind::Content))
            .map(|f| f.name)
            .collect();
        assert!(content.contains(&"payload_json"));
        assert!(content.contains(&"error_message"));
        assert!(content.contains(&"metadata"));
        assert!(content.contains(&"file_path"));
    }

    #[test]
    fn no_duplicate_names() {
        let names: Vec<&str> = ALL_COLUMNS.iter().map(|f| f.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len(), "duplicate column names found");
    }
}
