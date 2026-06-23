// Guard 2: the read crate cannot name the write crate's migration runner.
// `wqm-storage` does not depend on `wqm-storage-write`, so the path is unresolved.
fn main() {
    wqm_storage_write::migrations::run_migrations();
}
