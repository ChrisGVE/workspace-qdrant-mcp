//! Adaptive resource manager and background polling loop.
//!
//! Spawns a background task that monitors user activity and CPU load,
//! then communicates resource profile changes via a watch channel using
//! a 4-level state machine with gradual transitions.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use super::idle_detection::{is_cpu_under_pressure, seconds_since_last_input};
use super::{
    build_profiles, detect_physical_cores, log_startup_config, profile_for_level,
    AdaptiveResourceConfig, AdaptiveResourceState, Profiles, ResourceLevel, ResourceMode,
    ResourceProfile, SystemState,
};

/// Manages dynamic resource allocation based on system idle state.
///
/// Spawns a background polling task that monitors user activity and CPU load,
/// then communicates resource profile changes via a watch channel. Uses a
/// 4-level state machine (Normal < Active < Elevated < Burst) with gradual
/// transitions — max 1 level per evaluation, with configurable confirmation
/// delays for both ramp-up and ramp-down.
pub struct AdaptiveResourceManager {
    /// Receiver for the current resource profile
    rx: watch::Receiver<ResourceProfile>,
    /// Shared state for status reporting
    state: Arc<AdaptiveResourceState>,
}

impl AdaptiveResourceManager {
    /// Start the adaptive resource manager.
    ///
    /// Returns the manager (with a watch receiver) and spawns a background task
    /// that polls system state and updates the resource profile.
    ///
    /// `queue_depth` is an optional shared counter of pending queue items.
    /// When provided and > 0 while the user is active, the manager enters
    /// Active Processing mode with +50% resources.
    pub fn start(
        config: AdaptiveResourceConfig,
        cancellation_token: CancellationToken,
        queue_depth: Option<Arc<AtomicUsize>>,
    ) -> Self {
        let profiles = build_profiles(&config);
        let normal_profile = profiles.normal;

        let (tx, rx) = watch::channel(normal_profile);
        let state = Arc::new(AdaptiveResourceState::new());
        state.set_profile(&normal_profile);
        let state_clone = Arc::clone(&state);
        let physical_cores = detect_physical_cores();

        log_startup_config(&config, &profiles);

        tokio::spawn(async move {
            run_adaptive_loop(
                config,
                profiles,
                cancellation_token,
                tx,
                state_clone,
                physical_cores,
                queue_depth,
            )
            .await;
        });

        Self { rx, state }
    }

    /// Get the current resource profile.
    pub fn current_profile(&self) -> ResourceProfile {
        *self.rx.borrow()
    }

    /// Subscribe to resource profile changes.
    pub fn subscribe(&self) -> watch::Receiver<ResourceProfile> {
        self.rx.clone()
    }

    /// Get shared state handle for status reporting (gRPC/CLI).
    pub fn state(&self) -> Arc<AdaptiveResourceState> {
        Arc::clone(&self.state)
    }
}

/// Main adaptive resource loop — evaluates idle/active state and adjusts levels.
async fn run_adaptive_loop(
    config: AdaptiveResourceConfig,
    profiles: Profiles,
    cancellation_token: CancellationToken,
    tx: watch::Sender<ResourceProfile>,
    state: Arc<AdaptiveResourceState>,
    physical_cores: usize,
    queue_depth: Option<Arc<AtomicUsize>>,
) {
    let poll_interval = Duration::from_secs(config.poll_interval_secs);
    let idle_confirmation = Duration::from_secs(config.idle_confirmation_secs);
    let ramp_up_step = Duration::from_secs(config.ramp_up_step_secs);
    let ramp_down_step = Duration::from_secs(config.ramp_down_step_secs);
    let burst_hold = Duration::from_secs(config.burst_hold_secs);

    let mut sys_state = SystemState::new();
    let mut current_profile = profiles.normal;
    let mut heartbeat_counter: u64 = 0;
    let heartbeat_interval: u64 = 60 / config.poll_interval_secs.max(1);
    let mut mode_tracker = crate::idle_history::ModeTracker::new();
    let rotation_interval: u64 = 3600 / config.poll_interval_secs.max(1);

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                debug!("Adaptive resource manager shutting down");
                break;
            }
            _ = tokio::time::sleep(poll_interval) => {
                let idle_secs = seconds_since_last_input().unwrap_or(0.0);
                state.set_idle_seconds(idle_secs);

                let user_is_idle = idle_secs >= config.idle_threshold_secs as f64;
                let cpu_ok = !is_cpu_under_pressure(config.cpu_pressure_threshold, physical_cores);
                let old_level = sys_state.level;

                if user_is_idle && cpu_ok {
                    evaluate_ramp_up(&mut sys_state, &config, idle_secs, idle_confirmation, ramp_up_step);
                } else {
                    evaluate_ramp_down(&mut sys_state, &config, ramp_down_step, burst_hold);
                }

                // Active Processing Mode overlay: when user is present (not idle) and the
                // state machine is at Normal (no idle ramp-up), boost to Active profile if
                // the queue has pending work. The state machine level stays at Normal so
                // ramp-up/ramp-down logic is unaffected.
                let queue_has_work = queue_depth
                    .as_ref()
                    .map_or(false, |c| c.load(Ordering::Relaxed) > 0);
                let effective_level = if !user_is_idle
                    && sys_state.level == ResourceLevel::Normal
                    && queue_has_work
                {
                    ResourceLevel::Active
                } else {
                    sys_state.level
                };

                let new_profile = profile_for_level(
                    effective_level, &profiles.normal, &profiles.active,
                    &profiles.elevated, &profiles.burst,
                );
                let new_mode = ResourceMode::from(effective_level);
                state.set_mode(new_mode);
                state.set_profile(&new_profile);
                mode_tracker.on_mode_change(effective_level, idle_secs);

                if new_profile != current_profile {
                    if old_level != sys_state.level || effective_level != sys_state.level {
                        info!("Profile changed: embeddings {} -> {}, delay {}ms -> {}ms",
                            current_profile.max_concurrent_embeddings, new_profile.max_concurrent_embeddings,
                            current_profile.inter_item_delay_ms, new_profile.inter_item_delay_ms);
                    }
                    let _ = tx.send(new_profile);
                    current_profile = new_profile;
                }

                heartbeat_counter += 1;
                if heartbeat_counter % heartbeat_interval == 0 {
                    info!("Adaptive resources heartbeat: level={:?}, effective={:?}, mode={}, idle={:.0}s, cpu_pressure={}, embeddings={}, delay={}ms",
                        sys_state.level, effective_level, new_mode.as_str(), idle_secs, !cpu_ok,
                        new_profile.max_concurrent_embeddings, new_profile.inter_item_delay_ms);
                }
                if heartbeat_counter % rotation_interval == 0 {
                    mode_tracker.rotate();
                }
            }
        }
    }
}

/// Evaluate whether to ramp up resource levels during idle+CPU-ok state.
fn evaluate_ramp_up(
    sys_state: &mut SystemState,
    config: &AdaptiveResourceConfig,
    idle_secs: f64,
    idle_confirmation: Duration,
    ramp_up_step: Duration,
) {
    sys_state.activity_detected_at = None;

    if sys_state.idle_detected_at.is_none() {
        sys_state.idle_detected_at = Some(Instant::now());
        debug!(
            "Idle detected at level {:?}, starting confirmation ({}s)",
            sys_state.level, config.idle_confirmation_secs
        );
    }

    let idle_duration = sys_state
        .idle_detected_at
        .map(|t| t.elapsed())
        .unwrap_or_default();

    if idle_duration >= idle_confirmation && sys_state.level < ResourceLevel::Burst {
        let time_at_level = sys_state.level_entered_at.elapsed();
        if time_at_level >= ramp_up_step {
            let new_level = sys_state.level.up();
            info!(
                "Ramp-up: {:?} -> {:?} (idle {:.0}s, at level {:.0}s)",
                sys_state.level,
                new_level,
                idle_secs,
                time_at_level.as_secs_f64()
            );
            sys_state.transition_to(new_level);
        } else {
            debug!(
                "Ramp-up: holding at {:?} ({:.0}s/{:.0}s before next level)",
                sys_state.level,
                time_at_level.as_secs_f64(),
                ramp_up_step.as_secs_f64()
            );
        }
    } else if idle_duration < idle_confirmation {
        debug!(
            "Idle confirmation: {:.0}s/{:.0}s",
            idle_duration.as_secs_f64(),
            idle_confirmation.as_secs_f64()
        );
    }
}

/// Evaluate whether to ramp down resource levels during active/CPU-pressure state.
fn evaluate_ramp_down(
    sys_state: &mut SystemState,
    config: &AdaptiveResourceConfig,
    ramp_down_step: Duration,
    burst_hold: Duration,
) {
    sys_state.idle_detected_at = None;
    let target = ResourceLevel::Normal;

    if sys_state.level <= target {
        sys_state.activity_detected_at = None;
        return;
    }

    if sys_state.activity_detected_at.is_none() {
        sys_state.activity_detected_at = Some(Instant::now());
        debug!(
            "Activity detected at level {:?}, starting ramp-down ({}s/level)",
            sys_state.level, config.ramp_down_step_secs
        );
    }

    // Burst hold: enforce minimum time at burst before allowing ramp-down
    if sys_state.level == ResourceLevel::Burst {
        let burst_time = sys_state.level_entered_at.elapsed();
        if burst_time < burst_hold {
            debug!(
                "Burst hold: {:.0}s/{:.0}s before ramp-down allowed",
                burst_time.as_secs_f64(),
                burst_hold.as_secs_f64()
            );
            sys_state.activity_detected_at = None;
            return;
        }
    }

    if let Some(activity_start) = sys_state.activity_detected_at {
        let activity_duration = activity_start.elapsed();
        if activity_duration >= ramp_down_step {
            let new_level = sys_state.level.down();
            info!(
                "Ramp-down: {:?} -> {:?} (active {:.0}s)",
                sys_state.level,
                new_level,
                activity_duration.as_secs_f64()
            );
            sys_state.transition_to(new_level);
            sys_state.activity_detected_at = Some(Instant::now());
        } else {
            debug!(
                "Ramp-down: holding at {:?} ({:.0}s/{:.0}s before drop)",
                sys_state.level,
                activity_duration.as_secs_f64(),
                ramp_down_step.as_secs_f64()
            );
        }
    }
}
