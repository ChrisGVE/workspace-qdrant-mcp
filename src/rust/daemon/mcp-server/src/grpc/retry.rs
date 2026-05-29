//! Retry logic with exponential backoff for gRPC calls.
//!
//! Mirrors `DaemonClientBase.callWithRetry` from
//! `src/typescript/mcp-server/src/clients/daemon-client/connection.ts` lines 308-333,
//! and `isRetryableError` lines 335-354.
//!
//! # Policy (matches TS exactly)
//!
//! * Max 3 attempts (`MAX_RETRIES = 3`, TS line 32).
//! * Initial backoff delay: 100 ms (`INITIAL_RETRY_DELAY_MS = 100`, TS line 33).
//! * Delay doubles after each failed attempt: 100 ms → 200 ms.
//! * **No sleep after the last (3rd) attempt** — matches TS loop condition
//!   `if (attempt < this.maxRetries - 1) { await sleep(delay); delay *= 2; }`.
//! * Retryable gRPC status codes (by TYPE, not string matching — per RISK-8):
//!   - `tonic::Code::Unavailable`
//!   - `tonic::Code::DeadlineExceeded`
//!   - `tonic::Code::ResourceExhausted`
//!   - Plus transport / connection errors (tonic wraps these in `Status::unknown`
//!     or surfaces as `tonic::transport::Error`; see [`is_retryable`]).
//! * Non-retryable errors fail immediately without sleeping.
//!
//! # Testability seam
//!
//! The public `call_with_retry_inner` function accepts an injected async sleep
//! function (`sleep_fn: impl Fn(Duration) -> Fut`).  Tests supply an instant
//! no-op sleep and a closure that fails a controlled number of times.  The
//! production wrapper `call_with_retry` calls `call_with_retry_inner` with
//! `tokio::time::sleep`.

use std::time::Duration;
use tonic::Status;

/// Initial backoff delay in milliseconds (matches TS `INITIAL_RETRY_DELAY_MS = 100`).
pub const INITIAL_DELAY: Duration = Duration::from_millis(100);

/// Maximum number of attempts (matches TS `MAX_RETRIES = 3`).
pub const MAX_RETRIES: u32 = 3;

/// Classify a [`tonic::Status`] as retryable.
///
/// Uses code TYPE only — no string matching (RISK-8).
/// Mirrors TS `isRetryableError` lines 335-354.
pub fn is_retryable(status: &Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::Unavailable | tonic::Code::DeadlineExceeded | tonic::Code::ResourceExhausted
    ) || is_transport_error(status)
}

/// Detect tonic transport / connection errors.
///
/// Tonic surfaces transport errors as `Status` with code `Unknown` or `Internal`
/// when the channel drops, connection is refused, or the stream is reset.
/// We inspect the message for canonical indicators (matching TS lines 345-353).
fn is_transport_error(status: &Status) -> bool {
    let msg = status.message().to_ascii_lowercase();
    msg.contains("transport error")
        || msg.contains("connection refused")
        || msg.contains("connection reset")
        || msg.contains("broken pipe")
        || msg.contains("econnrefused")
        || msg.contains("etimedout")
        || msg.contains("enotfound")
        || msg.contains("channel has been shut down")
        || msg.contains("channel has been closed")
        || msg.contains("client not connected")
        // tonic wraps h2 stream errors this way
        || msg.contains("h2 protocol error")
}

/// Production retry wrapper — uses `tokio::time::sleep` for backoff delays.
///
/// # Errors
/// Returns the last error if all attempts are exhausted, or the first
/// non-retryable error immediately.
pub async fn call_with_retry<T, F, Fut>(f: F) -> Result<T, Status>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, Status>>,
{
    call_with_retry_inner(f, tokio::time::sleep).await
}

/// Testable retry core — sleep function is injected.
///
/// `sleep_fn` receives the backoff `Duration` to wait.  Tests pass a no-op
/// (`|_| async {}`) or a recording closure; production passes `tokio::time::sleep`.
///
/// Loop semantics (mirrors TS `callWithRetry` exactly):
/// ```text
/// attempt 0: call → if fail + retryable → sleep(100 ms) → delay *= 2
/// attempt 1: call → if fail + retryable → sleep(200 ms) → delay *= 2
/// attempt 2: call → if fail → NO sleep (attempt < maxRetries-1 is false) → return error
/// ```
pub async fn call_with_retry_inner<T, F, Fut, S, SFut>(f: F, sleep_fn: S) -> Result<T, Status>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, Status>>,
    S: Fn(Duration) -> SFut,
    SFut: std::future::Future<Output = ()>,
{
    let mut delay = INITIAL_DELAY;
    let mut last_error: Option<Status> = None;

    for attempt in 0..MAX_RETRIES {
        match f().await {
            Ok(value) => return Ok(value),
            Err(status) => {
                if !is_retryable(&status) {
                    // Non-retryable: fail immediately, no sleep.
                    return Err(status);
                }
                last_error = Some(status);
                // Sleep before next attempt, but NOT after the last attempt.
                if attempt < MAX_RETRIES - 1 {
                    sleep_fn(delay).await;
                    delay *= 2;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        Status::internal("call_with_retry: exhausted attempts with no recorded error")
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };
    use tonic::Code;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// No-op sleep: returns immediately.  Used to keep tests fast.
    async fn instant_sleep(_: Duration) {}

    /// Build a closure that fails with `code` for the first `fail_times` calls,
    /// then returns `Ok(())`.
    fn fail_n_times(
        fail_times: u32,
        code: Code,
    ) -> (
        Arc<AtomicU32>,
        impl Fn() -> std::future::Ready<Result<(), Status>>,
    ) {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let f = move || {
            let attempt = c.fetch_add(1, Ordering::SeqCst);
            if attempt < fail_times {
                std::future::ready(Err(Status::new(code, "injected error")))
            } else {
                std::future::ready(Ok(()))
            }
        };
        (counter, f)
    }

    // ── success paths ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn success_on_first_attempt_no_sleep() {
        // Counter tracks how many times sleep was called.
        let sleep_count = Arc::new(AtomicU32::new(0));
        let sc = sleep_count.clone();

        let (calls, f) = fail_n_times(0, Code::Unavailable);
        let result = call_with_retry_inner(f, move |_d| {
            sc.fetch_add(1, Ordering::SeqCst);
            async {}
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 1, "should call exactly once");
        assert_eq!(
            sleep_count.load(Ordering::SeqCst),
            0,
            "no sleep on immediate success"
        );
    }

    // ── retry on retryable codes ─────────────────────────────────────────────

    #[tokio::test]
    async fn retry_on_unavailable_success_on_third() {
        // Verify: exactly 2 sleeps (100 ms, 200 ms) and no trailing sleep.
        let sleep_durations: Arc<std::sync::Mutex<Vec<Duration>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let sd = sleep_durations.clone();

        let (calls, f) = fail_n_times(2, Code::Unavailable);
        let result = call_with_retry_inner(f, move |d| {
            sd.lock().unwrap().push(d);
            async {}
        })
        .await;

        assert!(result.is_ok(), "should succeed on 3rd attempt");
        assert_eq!(calls.load(Ordering::SeqCst), 3, "should call 3 times");

        let delays = sleep_durations.lock().unwrap().clone();
        assert_eq!(
            delays.len(),
            2,
            "exactly 2 sleeps (no sleep after last attempt)"
        );
        assert_eq!(
            delays[0],
            Duration::from_millis(100),
            "first backoff is 100 ms"
        );
        assert_eq!(
            delays[1],
            Duration::from_millis(200),
            "second backoff is 200 ms"
        );
    }

    #[tokio::test]
    async fn retry_on_deadline_exceeded() {
        let (calls, f) = fail_n_times(1, Code::DeadlineExceeded);
        let result = call_with_retry_inner(f, |_| async {}).await;
        assert!(result.is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_on_resource_exhausted() {
        let (calls, f) = fail_n_times(1, Code::ResourceExhausted);
        let result = call_with_retry_inner(f, |_| async {}).await;
        assert!(result.is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_on_transport_error() {
        // Transport errors come as Unknown with a characteristic message.
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();
        let f = move || {
            let c = cc.fetch_add(1, Ordering::SeqCst);
            let status = if c == 0 {
                Status::new(Code::Unknown, "transport error: connection refused")
            } else {
                return std::future::ready(Ok(()));
            };
            std::future::ready(Err(status))
        };

        let result = call_with_retry_inner(f, |_| async {}).await;
        assert!(result.is_ok());
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    // ── no retry on non-retryable codes ─────────────────────────────────────

    #[tokio::test]
    async fn no_retry_on_invalid_argument() {
        let sleep_count = Arc::new(AtomicU32::new(0));
        let sc = sleep_count.clone();
        let (calls, f) = fail_n_times(1, Code::InvalidArgument);
        let result = call_with_retry_inner(f, move |_| {
            sc.fetch_add(1, Ordering::SeqCst);
            async {}
        })
        .await;

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().code(),
            Code::InvalidArgument,
            "must surface original code"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "must not retry");
        assert_eq!(sleep_count.load(Ordering::SeqCst), 0, "must not sleep");
    }

    #[tokio::test]
    async fn no_retry_on_not_found() {
        let (calls, f) = fail_n_times(99, Code::NotFound);
        let result = call_with_retry_inner(f, |_| async {}).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), Code::NotFound);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn no_retry_on_unauthenticated() {
        let (calls, f) = fail_n_times(99, Code::Unauthenticated);
        let result = call_with_retry_inner(f, |_| async {}).await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn no_retry_on_permission_denied() {
        let (calls, f) = fail_n_times(99, Code::PermissionDenied);
        let result = call_with_retry_inner(f, |_| async {}).await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    // ── exhausts all 3 attempts and surfaces last error ──────────────────────

    #[tokio::test]
    async fn exhausts_all_attempts_surfaces_last_error() {
        // Always fail with UNAVAILABLE: should try 3 times total.
        let (calls, f) = fail_n_times(99, Code::Unavailable);
        let result = call_with_retry_inner(f, |_| async {}).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), Code::Unavailable);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            MAX_RETRIES,
            "must attempt exactly MAX_RETRIES times"
        );
    }

    #[tokio::test]
    async fn no_sleep_after_last_attempt() {
        // Three failures: sleep should fire exactly twice (after attempt 0 and 1).
        let sleep_count = Arc::new(AtomicU32::new(0));
        let sc = sleep_count.clone();
        let (_, f) = fail_n_times(99, Code::Unavailable);

        let _ = call_with_retry_inner(f, move |_| {
            sc.fetch_add(1, Ordering::SeqCst);
            async {}
        })
        .await;

        assert_eq!(
            sleep_count.load(Ordering::SeqCst),
            MAX_RETRIES - 1,
            "sleep count == MAX_RETRIES - 1 (no trailing sleep)"
        );
    }

    // ── backoff doubling ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn backoff_doubles_100ms_200ms() {
        // 3 failures → 2 sleeps at 100 ms and 200 ms.
        let delays: Arc<std::sync::Mutex<Vec<Duration>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let d = delays.clone();
        let (_, f) = fail_n_times(99, Code::Unavailable);

        let _ = call_with_retry_inner(f, move |dur| {
            d.lock().unwrap().push(dur);
            async {}
        })
        .await;

        let got = delays.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![Duration::from_millis(100), Duration::from_millis(200)]
        );
    }

    // ── is_retryable classifications ─────────────────────────────────────────

    #[test]
    fn unavailable_is_retryable() {
        assert!(is_retryable(&Status::new(Code::Unavailable, "")));
    }

    #[test]
    fn deadline_exceeded_is_retryable() {
        assert!(is_retryable(&Status::new(Code::DeadlineExceeded, "")));
    }

    #[test]
    fn resource_exhausted_is_retryable() {
        assert!(is_retryable(&Status::new(Code::ResourceExhausted, "")));
    }

    #[test]
    fn invalid_argument_not_retryable() {
        assert!(!is_retryable(&Status::new(Code::InvalidArgument, "")));
    }

    #[test]
    fn not_found_not_retryable() {
        assert!(!is_retryable(&Status::new(Code::NotFound, "")));
    }

    #[test]
    fn internal_not_retryable_generic() {
        // Generic Internal (not a transport error) is NOT retried.
        assert!(!is_retryable(&Status::new(
            Code::Internal,
            "some logic error"
        )));
    }

    #[test]
    fn transport_error_message_is_retryable() {
        let s = Status::new(Code::Unknown, "transport error: connection reset by peer");
        assert!(is_retryable(&s));
    }

    #[test]
    fn connection_refused_is_retryable() {
        let s = Status::new(Code::Unknown, "ECONNREFUSED 127.0.0.1:50051");
        assert!(is_retryable(&s));
    }

    #[test]
    fn channel_shut_down_is_retryable() {
        let s = Status::new(Code::Unknown, "Channel has been shut down");
        assert!(is_retryable(&s));
    }

    // ── constant sanity ──────────────────────────────────────────────────────

    #[test]
    fn initial_delay_is_100ms() {
        assert_eq!(INITIAL_DELAY, Duration::from_millis(100));
    }

    #[test]
    fn max_retries_is_3() {
        assert_eq!(MAX_RETRIES, 3);
    }
}
