//! Windows Service Control Manager (SCM) integration for memexd.
//!
//! When `memexd.exe` is launched by the SCM (`sc.exe start memexd` or
//! the equivalent management API), the binary connects back to the SCM
//! via the windows-service crate's `service_dispatcher::start` and runs
//! `service_main` in a dedicated thread.
//!
//! When the binary is launched interactively (foreground / from a
//! shell), the dispatcher call returns
//! `Error::Winapi(ERROR_FAILED_SERVICE_CONTROLLER_CONNECT)` and we fall
//! through to the normal CLI startup path.
//!
//! This module is compiled only on `cfg(windows)`. It exposes:
//! - [`try_start_dispatcher`]: best-effort connect to SCM, returning
//!   `true` when the dispatch ran (the service has started + stopped).
//! - [`scm_shutdown_signal`]: a `Notify` the rest of the daemon can
//!   `await` to receive an SCM Stop / Shutdown control code.

#![cfg(windows)]

use std::ffi::OsString;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::Notify;
use tracing::{error, info, warn};

use windows_service::define_windows_service;
use windows_service::service::{
    ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
use windows_service::service_dispatcher;

/// Service name registered with the Windows SCM. Must match the name
/// used by `wqm service install` on Windows.
pub const SERVICE_NAME: &str = "memexd";

/// Service type for memexd. `OWN_PROCESS` means the binary is a
/// stand-alone service (one process per service).
const SERVICE_TYPE: ServiceType = ServiceType::OWN_PROCESS;

/// Lazily-initialised Notify used to forward an SCM Stop or Shutdown
/// control code to the async daemon's shutdown waiter.
fn shutdown_notify() -> &'static Arc<Notify> {
    static NOTIFY: OnceLock<Arc<Notify>> = OnceLock::new();
    NOTIFY.get_or_init(|| Arc::new(Notify::new()))
}

/// Public handle the daemon awaits to detect an SCM-initiated stop.
pub fn scm_shutdown_signal() -> Arc<Notify> {
    Arc::clone(shutdown_notify())
}

/// Try to connect to the SCM and run as a service.
///
/// Returns `true` when the dispatcher ran to completion (the service
/// has started and stopped — the process should exit). Returns `false`
/// when the binary was not started by the SCM (interactive mode), so
/// the caller should continue with the normal CLI startup path.
pub fn try_start_dispatcher() -> bool {
    match service_dispatcher::start(SERVICE_NAME, ffi_service_main) {
        Ok(()) => true,
        Err(windows_service::Error::Winapi(io_err)) if io_err.raw_os_error() == Some(1063) => {
            // ERROR_FAILED_SERVICE_CONTROLLER_CONNECT — not started by
            // the SCM. Fall through to interactive.
            false
        }
        Err(e) => {
            error!("service_dispatcher::start failed: {}", e);
            // Treat any other error as a fall-through too — better to
            // run interactively than to refuse to start at all.
            false
        }
    }
}

// `define_windows_service!` generates the FFI shim
// `extern "system" fn ffi_service_main(argc, argv)` that the SCM calls.
// It collects the args into `Vec<OsString>` and invokes
// `service_main`.
define_windows_service!(ffi_service_main, service_main);

/// SCM service entry point.
///
/// Registers a control handler, transitions the SCM status through
/// StartPending → Running, runs the async daemon to completion, then
/// transitions Stopped.
fn service_main(_args: Vec<OsString>) {
    let status_handle = match register_handler_and_start_pending() {
        Some(h) => h,
        None => return,
    };

    let rt = match build_tokio_runtime(&status_handle) {
        Some(rt) => rt,
        None => return,
    };

    run_daemon_under_scm(rt, status_handle);
}

/// Register the SCM control handler and signal StartPending.
/// Returns `None` if registration fails (caller should return).
fn register_handler_and_start_pending(
) -> Option<windows_service::service_control_handler::ServiceStatusHandle> {
    let notify = scm_shutdown_signal();
    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                info!("SCM control code: {:?}; signalling shutdown", control_event);
                notify.notify_waiters();
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };

    let status_handle = match service_control_handler::register(SERVICE_NAME, event_handler) {
        Ok(h) => h,
        Err(e) => {
            error!("Failed to register SCM control handler: {}", e);
            return None;
        }
    };

    if let Err(e) = status_handle.set_service_status(ServiceStatus {
        service_type: SERVICE_TYPE,
        current_state: ServiceState::StartPending,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 1,
        wait_hint: Duration::from_secs(30),
        process_id: None,
    }) {
        warn!("Failed to set StartPending status: {}", e);
    }
    Some(status_handle)
}

/// Build multi-threaded tokio runtime; mark service Stopped and return `None` on failure.
fn build_tokio_runtime(
    status_handle: &windows_service::service_control_handler::ServiceStatusHandle,
) -> Option<tokio::runtime::Runtime> {
    match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => Some(rt),
        Err(e) => {
            error!("Failed to build tokio runtime for SCM service: {}", e);
            let _ = status_handle.set_service_status(ServiceStatus {
                service_type: SERVICE_TYPE,
                current_state: ServiceState::Stopped,
                controls_accepted: ServiceControlAccept::empty(),
                exit_code: ServiceExitCode::Win32(1),
                checkpoint: 0,
                wait_hint: Duration::ZERO,
                process_id: None,
            });
            None
        }
    }
}

/// Set Running, block on the async daemon, then set Stopped.
fn run_daemon_under_scm(
    rt: tokio::runtime::Runtime,
    status_handle: windows_service::service_control_handler::ServiceStatusHandle,
) {
    if let Err(e) = status_handle.set_service_status(ServiceStatus {
        service_type: SERVICE_TYPE,
        current_state: ServiceState::Running,
        controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::ZERO,
        process_id: None,
    }) {
        warn!("Failed to set Running status: {}", e);
    }

    let daemon_result = rt.block_on(async {
        let args = match crate::startup::parse_args() {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let config = match crate::startup::load_config(&args) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        crate::startup::init_logging_with_telemetry(
            &args.log_level,
            false,
            Some(&config.observability.telemetry),
        )?;
        info!("memexd running under Windows SCM");
        crate::run_daemon(config, args).await
    });

    let exit = match &daemon_result {
        Ok(()) => 0,
        Err(e) => {
            error!("Daemon exited with error under SCM: {}", e);
            1
        }
    };

    if let Err(e) = status_handle.set_service_status(ServiceStatus {
        service_type: SERVICE_TYPE,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(exit),
        checkpoint: 0,
        wait_hint: Duration::ZERO,
        process_id: None,
    }) {
        warn!("Failed to set Stopped status: {}", e);
    }
}
