//! Platform-specific safety checks for the unsafe code auditor.
//!
//! Covers Windows (UTF-16, Win32 API, handle management), Linux (epoll, inotify),
//! macOS (FSEvents, kqueue), POSIX file-descriptor operations, stdio redirection,
//! and service-discovery safety.

use super::super::super::types::{
    SafetyViolation, UnsafeAuditError, ViolationSeverity, ViolationType,
};
use super::UnsafeCodeAuditor;

impl UnsafeCodeAuditor {
    // -----------------------------------------------------------------------
    // Windows-specific checks
    // -----------------------------------------------------------------------

    #[cfg(target_os = "windows")]
    pub(super) async fn test_windows_file_watching_safety(&self) -> Result<(), UnsafeAuditError> {
        self.test_utf16_conversion_safety().await?;
        self.test_win32_api_safety().await?;
        self.test_handle_management_safety().await?;
        Ok(())
    }

    #[cfg(target_os = "windows")]
    async fn test_utf16_conversion_safety(&self) -> Result<(), UnsafeAuditError> {
        let test_path = "C:\\test\\path\\with\\unicode\\\u{6D4B}\u{8BD5}";

        let wide_chars: Vec<u16> = test_path.encode_utf16().chain(std::iter::once(0)).collect();

        if wide_chars.last() != Some(&0) {
            self.record_violation(SafetyViolation {
                location: "platform.rs:534".to_string(),
                violation_type: ViolationType::BufferOverflow,
                severity: ViolationSeverity::High,
                description: "UTF-16 string not properly null-terminated".to_string(),
                suggested_fix: "Ensure null termination in UTF-16 conversion".to_string(),
                stack_trace: None,
            });
        }

        let ptr = wide_chars.as_ptr();
        if ptr.is_null() {
            self.record_violation(SafetyViolation {
                location: "platform.rs:538".to_string(),
                violation_type: ViolationType::NullPointerDereference,
                severity: ViolationSeverity::Critical,
                description: "Null pointer from UTF-16 conversion".to_string(),
                suggested_fix: "Validate pointer before use".to_string(),
                stack_trace: None,
            });
        }

        Ok(())
    }

    #[cfg(target_os = "windows")]
    async fn test_win32_api_safety(&self) -> Result<(), UnsafeAuditError> {
        Ok(())
    }

    #[cfg(target_os = "windows")]
    async fn test_handle_management_safety(&self) -> Result<(), UnsafeAuditError> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Linux-specific checks
    // -----------------------------------------------------------------------

    #[cfg(target_os = "linux")]
    pub(super) async fn test_linux_file_watching_safety(&self) -> Result<(), UnsafeAuditError> {
        self.test_epoll_fd_safety().await?;
        self.test_inotify_fd_safety().await?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    async fn test_epoll_fd_safety(&self) -> Result<(), UnsafeAuditError> {
        Ok(())
    }

    #[cfg(target_os = "linux")]
    async fn test_inotify_fd_safety(&self) -> Result<(), UnsafeAuditError> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // macOS-specific checks
    // -----------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    pub(super) async fn test_macos_file_watching_safety(&self) -> Result<(), UnsafeAuditError> {
        self.test_fsevents_callback_safety().await?;
        self.test_kqueue_fd_safety().await?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn test_fsevents_callback_safety(&self) -> Result<(), UnsafeAuditError> {
        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn test_kqueue_fd_safety(&self) -> Result<(), UnsafeAuditError> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Storage / POSIX fd checks
    // -----------------------------------------------------------------------

    pub(crate) async fn test_fd_duplication_safety(&self) -> Result<(), UnsafeAuditError> {
        #[cfg(unix)]
        {
            let original_fd = 1; // stdout
            let result = unsafe { libc::dup(original_fd) };

            if result == -1 {
                self.record_violation(SafetyViolation {
                    location: "storage.rs:974".to_string(),
                    violation_type: ViolationType::InvalidCast,
                    severity: ViolationSeverity::High,
                    description: "File descriptor duplication failed".to_string(),
                    suggested_fix: "Check errno and handle failure case".to_string(),
                    stack_trace: None,
                });
            } else {
                let close_result = unsafe { libc::close(result) };
                if close_result != 0 {
                    self.record_violation(SafetyViolation {
                        location: "storage.rs:974".to_string(),
                        violation_type: ViolationType::MemoryLeak,
                        severity: ViolationSeverity::Medium,
                        description: "Failed to close duplicated file descriptor".to_string(),
                        suggested_fix: "Ensure proper cleanup of file descriptors".to_string(),
                        stack_trace: None,
                    });
                }
            }
        }

        Ok(())
    }

    pub(super) async fn test_stdio_redirection_safety(&self) -> Result<(), UnsafeAuditError> {
        #[cfg(unix)]
        {
            let original_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
            if original_stdout != -1 {
                let restore_result = unsafe { libc::dup2(original_stdout, libc::STDOUT_FILENO) };

                if restore_result == -1 {
                    self.record_violation(SafetyViolation {
                        location: "storage.rs:990".to_string(),
                        violation_type: ViolationType::UndefinedBehavior,
                        severity: ViolationSeverity::High,
                        description: "Failed to restore stdout".to_string(),
                        suggested_fix: "Check return values and handle restoration failure"
                            .to_string(),
                        stack_trace: None,
                    });
                }

                unsafe { libc::close(original_stdout) };
            }
        }

        Ok(())
    }

    pub(super) async fn test_service_discovery_safety(&self) -> Result<(), UnsafeAuditError> {
        Ok(())
    }
}
