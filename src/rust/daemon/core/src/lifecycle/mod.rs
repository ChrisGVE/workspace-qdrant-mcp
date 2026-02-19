//! State Machine: watch folder and project lifecycle management.
//!
//! Centralizes all watch_folders state mutations to fix the split-brain
//! where DaemonStateManager, PriorityManager, and CLI all mutate
//! `watch_folders.is_active` independently.
//!
//! # Future submodules
//! - `watch_folder` — WatchFolderLifecycle state machine
//! - `project` — project lifecycle (register/activate/deactivate)

// Submodules will be added in Phase 3.
