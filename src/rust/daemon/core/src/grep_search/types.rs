/// File info from file_metadata needed for grep scanning.
#[derive(Debug, Clone)]
pub(super) struct FileInfo {
    pub(super) file_path: String,
    pub(super) tenant_id: String,
    pub(super) branch: Option<String>,
}
