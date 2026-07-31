//! What the system's health is, and what a change in it is allowed to announce.
//!
//! The toast widget renders an alert; this module decides whether there is one. Keeping the
//! two apart matters because the interesting rules are all here — *which* components may
//! raise an alarm, and when one must stay quiet — and none of them are rendering questions.
//!
//! # The rule covers every component, and the daemon (Chris, 20260731)
//!
//! Not four transitions and not one store: **every component the report names, plus the
//! daemon**, which is on or off. `store_health` already renders whatever list arrives
//! (`CR-036`(a) — the component set is data, not a UI constant), and this follows it: a
//! component this build has never heard of transitions exactly like one it has.
//!
//! # The daemon is the liveness master, so it silences the others
//!
//! VISUAL-LANGUAGE §7 makes the daemon the liveness master, and every component's health is
//! observed *through* it. So when the daemon goes unreachable, the components do not each
//! raise an alarm — their state is **unknown**, not offline, and reporting four "store
//! offline" alerts for one daemon failure is precisely the dilution Chris cut this surface
//! down to avoid. One alarm fires: the daemon's. While it stays down, nothing else does.
//!
//! On the way back the balance flips: the daemon's recovery is one all-clear, and a component
//! that is **still unhealthy once things have settled** raises its own alarm, because that
//! genuinely is news the all-clear would otherwise hide. A component that comes back healthy
//! raises nothing — the daemon's all-clear already said so.
//!
//! # A returning daemon gets a settling window (Chris, 20260731)
//!
//! *"When the daemon is back, it would require some time to ascertain the state of all its
//! resources."* A daemon that has just come up has not finished probing Qdrant, the graph
//! engine or its queue — its first readings describe **its own startup**, not the system. So
//! [`HealthWatch`] withholds component alarms for [`SETTLE_AFTER_RECOVERY`] after a recovery,
//! and when the window closes it judges the **state**, not the churn: whatever is still
//! unhealthy alarms once, and a component that broke and healed inside the window never
//! happened as far as the user is concerned.
//!
//! This is the one debounce that lives on this side. General flapping stays daemon-side under
//! `CR-035` (`HEALTH-MONITORING.md` property 3) and always will — the difference is that this
//! window is not trying to detect a settled change, it is declining to trust a source that has
//! told us it is not ready.
//!
//! # The sustained half is not a toast
//!
//! A toast announces a transition and expires. "The daemon is unreachable" has to stay said
//! for as long as it is true, which is [`crate::tokens::Condition`] and the red wash
//! ([`crate::widgets::surface`]). [`SystemHealth::condition`] is where the one state becomes
//! the other, so the wash and the toast can never disagree about whether the daemon is up.

use std::time::{Duration, Instant};

use wqm_client::{DaemonReport, DaemonState, UnreachableReason};

use crate::tokens::{Condition, Health};
use crate::widgets::toast::Toast;

/// One component of the system as the report names it.
///
/// `role` is an owned string for the same reason `store_health::StoreRow::role` is: the set is
/// data that arrives from the wire, and a fixed enum here would need a UI change per component
/// (`CR-036`(a)). It is **not** contract-bound — `CR-036`(b) is the open half, and until it is
/// answered this module names no component itself.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Component {
    pub role: String,
    pub health: Health,
}

impl Component {
    pub fn new(role: impl Into<String>, health: Health) -> Self {
        Self {
            role: role.into(),
            health,
        }
    }
}

/// The whole system's health at one moment: the daemon, and whatever it reported.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SystemHealth {
    /// The liveness master. [`Health::Offline`] is "unreachable" — N49's two-variant
    /// `DaemonReport` is exactly this on/off distinction, with `Degraded` reserved for a
    /// daemon that answered with a claim this build cannot read.
    pub daemon: Health,
    pub components: Vec<Component>,
}

impl SystemHealth {
    pub fn new(daemon: Health, components: Vec<Component>) -> Self {
        Self { daemon, components }
    }

    /// The system as N49 reports it, with the component list the caller already holds.
    ///
    /// The daemon mapping is `daemon_status::health_of`'s, restated in one place rather than
    /// re-derived: `Unrecognized` is degraded, never healthy.
    pub fn from_report(report: &DaemonReport, components: Vec<Component>) -> Self {
        let daemon = match report {
            DaemonReport::Reachable(status) => match status.state {
                DaemonState::Ok => Health::Healthy,
                DaemonState::Unrecognized(_) => Health::Degraded,
            },
            DaemonReport::Unreachable { .. } => Health::Offline,
        };
        Self::new(daemon, components)
    }

    /// The screen-wide condition this state implies. The single place the wash is decided, so
    /// it cannot disagree with the toasts.
    pub fn condition(&self) -> Condition {
        match self.daemon {
            Health::Offline => Condition::DaemonUnreachable,
            _ => Condition::Nominal,
        }
    }

    fn component(&self, role: &str) -> Option<Health> {
        self.components
            .iter()
            .find(|c| c.role == role)
            .map(|c| c.health)
    }
}

/// The word for arriving at a state. `Healthy` is *recovered* rather than *healthy* because a
/// toast is always about a change — nothing here is ever a status line.
const fn verb(health: Health) -> &'static str {
    match health {
        Health::Healthy => "recovered",
        Health::Degraded => "degraded",
        Health::Offline => "offline",
    }
}

/// The daemon's own offline wording, taken from N49 rather than written here, so the screen
/// says what the wire says (MCP-SURFACE §4.5 rule 1).
fn daemon_offline_label() -> String {
    UnreachableReason::DaemonUnreachable
        .as_str()
        .replace('_', " ")
}

fn daemon_message(to: Health) -> String {
    match to {
        Health::Offline => daemon_offline_label(),
        other => format!("daemon {}", verb(other)),
    }
}

/// How long after the daemon returns component alarms are withheld.
///
/// Chris, 20260731: *"when the daemon is back, it would require some time to ascertain the
/// state of all its resources."* A daemon that has just come up has not finished probing
/// Qdrant, the graph engine or its queue, so its first reports are provisional — alarming on
/// them announces the daemon's own startup rather than anything wrong with the system.
///
/// A *tolerance*, so provisional and not this module's to set: requested as an N7 knob in
/// `UIQ-008`, and named here with the same word the request uses.
pub const SETTLE_AFTER_RECOVERY: Duration = Duration::from_secs(5);

/// The alert policy with memory: the diff below, plus the settling window the diff cannot
/// express.
///
/// [`transitions`] is a pure function of two observations, which is what makes it testable —
/// but "wait a few seconds before believing this" is a property of *time*, not of a pair of
/// states. So the window lives here, and, per the rule the whole surface is built on, this
/// type never reads the clock either: [`HealthWatch::observe`] takes `now`.
#[derive(Clone, Debug)]
pub struct HealthWatch {
    last: Option<SystemHealth>,
    settling_until: Option<Instant>,
    settle_window: Duration,
}

impl Default for HealthWatch {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthWatch {
    pub fn new() -> Self {
        Self::with_settle_window(SETTLE_AFTER_RECOVERY)
    }

    pub fn with_settle_window(settle_window: Duration) -> Self {
        Self {
            last: None,
            settling_until: None,
            settle_window,
        }
    }

    /// The condition the last observation implies — [`Condition::Nominal`] before anything has
    /// been observed, because an unobserved daemon is not a failed one.
    pub fn condition(&self) -> Condition {
        self.last
            .as_ref()
            .map(SystemHealth::condition)
            .unwrap_or(Condition::Nominal)
    }

    /// Whether component alarms are currently being withheld.
    pub fn is_settling(&self, now: Instant) -> bool {
        self.settling_until.is_some_and(|until| now < until)
    }

    /// Take an observation and return the toasts it is allowed to raise.
    ///
    /// Four behaviours worth stating plainly, because each one is a deliberate silence:
    ///
    /// 1. **The first observation says nothing.** It is a baseline, not a change — a TUI opened
    ///    onto an already-degraded system announces nothing, it *shows* it.
    /// 2. **A daemon outage silences its components**, exactly as [`transitions`] does.
    /// 3. **The window after a recovery withholds component alarms.** During it, component
    ///    churn is tracked and never announced.
    /// 4. **When the window closes, the state is judged, not the churn.** Whatever is still
    ///    unhealthy alarms once, at that moment. A component that broke and healed inside the
    ///    window never existed as far as the user is concerned.
    ///
    /// The window closes on the first observation *after* it expires — this type has no timer
    /// of its own, and a status stream that stopped arriving has a bigger problem than a late
    /// toast.
    pub fn observe(&mut self, next: SystemHealth, now: Instant) -> Vec<Toast> {
        let Some(before) = self.last.replace(next.clone()) else {
            // Baseline. A first look is not news.
            return Vec::new();
        };

        let mut toasts = Vec::new();
        if before.daemon != next.daemon {
            toasts.extend(Toast::transition(
                before.daemon,
                next.daemon,
                daemon_message(next.daemon),
            ));
        }

        if next.daemon == Health::Offline {
            // A fresh outage ends any settling: there is nothing to settle toward.
            self.settling_until = None;
            return toasts;
        }

        if before.daemon == Health::Offline {
            // Just back. The all-clear goes out now; the components get their grace.
            self.settling_until = Some(now + self.settle_window);
            return toasts;
        }

        match self.settling_until {
            Some(until) if now < until => toasts,
            Some(_) => {
                self.settling_until = None;
                // The state at the end of the window is the claim worth making. Comparing
                // against `before` here would announce whatever the daemon's last provisional
                // reading happened to be, which is the noise the window exists to remove.
                toasts.extend(unhealthy_alarms(&next));
                toasts
            }
            None => {
                toasts.extend(component_transitions(&before, &next));
                toasts
            }
        }
    }
}

/// One alarm per component that is not healthy, as a statement about *now* rather than about a
/// change. Used only when the settling window closes.
fn unhealthy_alarms(state: &SystemHealth) -> Vec<Toast> {
    state
        .components
        .iter()
        .filter(|component| component.health != Health::Healthy)
        .filter_map(|component| {
            Toast::transition(
                Health::Healthy,
                component.health,
                format!("{} {}", component.role, verb(component.health)),
            )
        })
        .collect()
}

/// Every toast that a move from `before` to `after` is allowed to raise, in the order they
/// should be pushed.
///
/// The diff half of the policy. [`HealthWatch`] is the entry point a screen uses — it wraps
/// this with the settling window — and a caller that pushed its own toasts instead would be
/// able to bypass both.
pub fn transitions(before: &SystemHealth, after: &SystemHealth) -> Vec<Toast> {
    let mut toasts = Vec::new();

    let daemon_changed = before.daemon != after.daemon;
    if daemon_changed {
        toasts.extend(Toast::transition(
            before.daemon,
            after.daemon,
            daemon_message(after.daemon),
        ));
    }

    // Down, or going down: the daemon's alarm is the only one. Every component reading is
    // unknown rather than offline, and four alarms for one failure is the dilution this
    // surface exists to avoid.
    if after.daemon == Health::Offline {
        return toasts;
    }

    if before.daemon == Health::Offline {
        // Coming back, with no clock available here: report what is unhealthy right away.
        // [`HealthWatch`] is the version that waits first, and it is what a screen uses —
        // this branch is what "the diff alone would have said".
        toasts.extend(unhealthy_alarms(after));
        return toasts;
    }

    toasts.extend(component_transitions(before, after));
    toasts
}

/// The component half, with both observations taken through a live daemon.
///
/// A component that disappeared from the report says nothing: its absence is not a state, and
/// `store_health` already shows a vanished row as unreadable rather than dropping it.
fn component_transitions(before: &SystemHealth, after: &SystemHealth) -> Vec<Toast> {
    after
        .components
        .iter()
        .filter_map(|component| {
            let previous = before.component(&component.role);
            let raise = match previous {
                // A component that was already known: only a real change speaks.
                Some(was) => was != component.health,
                // One that has just appeared: news only if it arrives in trouble.
                None => component.health != Health::Healthy,
            };
            if !raise {
                return None;
            }

            // A component with no previous reading is treated as having been healthy, so
            // `transition` still describes a change and cannot be handed the same state twice.
            let from = previous.unwrap_or(Health::Healthy);
            Toast::transition(
                from,
                component.health,
                format!("{} {}", component.role, verb(component.health)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn system(daemon: Health, components: &[(&str, Health)]) -> SystemHealth {
        SystemHealth::new(
            daemon,
            components
                .iter()
                .map(|(role, health)| Component::new(*role, *health))
                .collect(),
        )
    }

    fn messages(toasts: &[Toast]) -> Vec<&str> {
        toasts.iter().map(|t| t.message()).collect()
    }

    const NOMINAL: [(&str, Health); 3] = [
        ("vector", Health::Healthy),
        ("graph", Health::Healthy),
        ("relational", Health::Healthy),
    ];

    #[test]
    fn every_component_transitions_not_just_a_chosen_few() {
        let before = system(Health::Healthy, &NOMINAL);
        let after = system(
            Health::Healthy,
            &[
                ("vector", Health::Degraded),
                ("graph", Health::Offline),
                ("relational", Health::Healthy),
            ],
        );
        // Two changed, one did not. The set is whatever the report carries — a component this
        // build never heard of behaves identically, which CR-036(a) requires.
        assert_eq!(
            messages(&transitions(&before, &after)),
            vec!["vector degraded", "graph offline"]
        );

        let arrivals = system(
            Health::Healthy,
            &[
                ("vector", Health::Healthy),
                ("graph", Health::Healthy),
                ("relational", Health::Healthy),
                ("embedding_provider", Health::Degraded),
                ("queue_processor", Health::Healthy),
            ],
        );
        assert_eq!(
            messages(&transitions(&before, &arrivals)),
            vec!["embedding_provider degraded"],
            "a new component speaks only if it arrives in trouble"
        );
    }

    #[test]
    fn the_daemon_going_unreachable_silences_every_component() {
        let before = system(Health::Healthy, &NOMINAL);
        // The stores "go offline" in the same tick — which is what losing the daemon looks
        // like from here, and exactly the case that would otherwise fire four alarms.
        let after = system(
            Health::Offline,
            &[
                ("vector", Health::Offline),
                ("graph", Health::Offline),
                ("relational", Health::Offline),
            ],
        );

        let toasts = transitions(&before, &after);
        assert_eq!(
            messages(&toasts),
            vec!["daemon unreachable"],
            "one failure must produce one alarm"
        );
        assert!(!toasts[0].is_recovery());
        // And the sustained half is the wash, not a toast.
        assert_eq!(after.condition(), Condition::DaemonUnreachable);
    }

    #[test]
    fn nothing_speaks_while_the_daemon_stays_down() {
        let down = system(Health::Offline, &[("vector", Health::Offline)]);
        let still_down = system(
            Health::Offline,
            &[("vector", Health::Degraded), ("graph", Health::Offline)],
        );
        assert!(
            transitions(&down, &still_down).is_empty(),
            "readings taken through a dead daemon are not events"
        );
    }

    #[test]
    fn coming_back_says_the_all_clear_and_only_the_bad_news_under_it() {
        let down = system(Health::Offline, &[]);
        let back = system(
            Health::Healthy,
            &[
                ("vector", Health::Healthy),
                ("graph", Health::Degraded),
                ("relational", Health::Healthy),
            ],
        );

        let toasts = transitions(&down, &back);
        assert_eq!(
            messages(&toasts),
            vec!["daemon recovered", "graph degraded"],
            "the all-clear covers what came back healthy; what did not still speaks"
        );
        assert!(toasts[0].is_recovery());
        assert!(!toasts[1].is_recovery());
        assert_eq!(back.condition(), Condition::Nominal);
    }

    #[test]
    fn a_steady_system_says_nothing_however_often_it_is_polled() {
        let state = system(Health::Healthy, &NOMINAL);
        for _ in 0..8 {
            assert!(
                transitions(&state, &state).is_empty(),
                "polling is not an event"
            );
        }

        // Including a steadily unhappy one: a component that stays degraded has already been
        // announced once, and repeating it is how an alert loses its meaning.
        let degraded = system(Health::Healthy, &[("vector", Health::Degraded)]);
        assert!(transitions(&degraded, &degraded).is_empty());
    }

    #[test]
    fn a_daemon_that_answers_with_a_state_we_cannot_read_is_degraded_not_offline() {
        // N49's own warning, and the reason the daemon is not a plain boolean here: it is
        // reachable, so its components are still observable and the screen is not washed.
        let before = system(Health::Healthy, &NOMINAL);
        let after = system(Health::Degraded, &NOMINAL);

        assert_eq!(
            messages(&transitions(&before, &after)),
            vec!["daemon degraded"]
        );
        assert_eq!(after.condition(), Condition::Nominal);
    }

    #[test]
    fn a_vanished_component_says_nothing() {
        let before = system(Health::Healthy, &NOMINAL);
        let after = system(Health::Healthy, &[("vector", Health::Healthy)]);
        assert!(
            transitions(&before, &after).is_empty(),
            "an absent reading is not a state, and CR-036(a) puts the unreadable case in the row"
        );
    }

    #[test]
    fn a_first_observation_is_a_baseline_and_never_news() {
        let now = Instant::now();
        let mut watch = HealthWatch::new();
        // A TUI opened onto an already-degraded system shows it; it does not announce it.
        let toasts = watch.observe(
            system(Health::Healthy, &[("vector", Health::Degraded)]),
            now,
        );
        assert!(toasts.is_empty());
        assert_eq!(watch.condition(), Condition::Nominal);
    }

    #[test]
    fn a_returning_daemon_gets_a_settling_window_before_its_components_may_alarm() {
        let start = Instant::now();
        let window = Duration::from_secs(5);
        let mut watch = HealthWatch::with_settle_window(window);

        watch.observe(system(Health::Healthy, &NOMINAL), start);
        let down = watch.observe(system(Health::Offline, &[]), start);
        assert_eq!(messages(&down), vec!["daemon unreachable"]);

        // Back, with a store still reporting trouble in the first breath. The all-clear goes
        // out; the store does not, because the daemon has not finished probing it.
        let back = watch.observe(
            system(
                Health::Healthy,
                &[("vector", Health::Offline), ("graph", Health::Healthy)],
            ),
            start,
        );
        assert_eq!(messages(&back), vec!["daemon recovered"]);
        assert!(watch.is_settling(start));

        // Mid-window churn is tracked and stays silent — this is the noise the window removes.
        let mid = watch.observe(
            system(
                Health::Healthy,
                &[("vector", Health::Degraded), ("graph", Health::Degraded)],
            ),
            start + Duration::from_secs(2),
        );
        assert!(mid.is_empty(), "the window leaked: {:?}", messages(&mid));
    }

    #[test]
    fn when_the_window_closes_the_state_is_judged_not_the_churn() {
        let start = Instant::now();
        let window = Duration::from_secs(5);
        let mut watch = HealthWatch::with_settle_window(window);

        watch.observe(system(Health::Healthy, &NOMINAL), start);
        watch.observe(system(Health::Offline, &[]), start);
        watch.observe(
            system(
                Health::Healthy,
                &[("vector", Health::Offline), ("graph", Health::Offline)],
            ),
            start,
        );

        // `graph` broke and healed inside the window: as far as the user is concerned it never
        // happened. `vector` is still down when the window closes, so it alarms — once.
        let settled = watch.observe(
            system(
                Health::Healthy,
                &[("vector", Health::Degraded), ("graph", Health::Healthy)],
            ),
            start + window,
        );
        assert_eq!(messages(&settled), vec!["vector degraded"]);
        assert!(!watch.is_settling(start + window));

        // And afterwards the ordinary diff is back in force: no repeat for an unchanged state.
        let steady = watch.observe(
            system(
                Health::Healthy,
                &[("vector", Health::Degraded), ("graph", Health::Healthy)],
            ),
            start + window + Duration::from_secs(1),
        );
        assert!(steady.is_empty());
    }

    #[test]
    fn a_system_that_comes_back_clean_says_only_the_all_clear() {
        let start = Instant::now();
        let window = Duration::from_secs(5);
        let mut watch = HealthWatch::with_settle_window(window);

        watch.observe(system(Health::Healthy, &NOMINAL), start);
        watch.observe(system(Health::Offline, &[]), start);
        watch.observe(system(Health::Healthy, &NOMINAL), start);

        let settled = watch.observe(system(Health::Healthy, &NOMINAL), start + window);
        assert!(
            settled.is_empty(),
            "a clean recovery must not speak twice: {:?}",
            messages(&settled)
        );
    }

    #[test]
    fn a_second_outage_during_the_window_cancels_it() {
        let start = Instant::now();
        let window = Duration::from_secs(5);
        let mut watch = HealthWatch::with_settle_window(window);

        watch.observe(system(Health::Healthy, &NOMINAL), start);
        watch.observe(system(Health::Offline, &[]), start);
        watch.observe(system(Health::Healthy, &NOMINAL), start);

        // A flapping daemon: down again before its components ever settled.
        let down_again =
            watch.observe(system(Health::Offline, &[]), start + Duration::from_secs(1));
        assert_eq!(messages(&down_again), vec!["daemon unreachable"]);
        assert!(
            !watch.is_settling(start + Duration::from_secs(2)),
            "there is nothing to settle toward while it is down"
        );

        // ...and the next recovery starts a fresh window rather than inheriting the old one.
        watch.observe(
            system(Health::Healthy, &[("vector", Health::Degraded)]),
            start + Duration::from_secs(3),
        );
        assert!(watch.is_settling(start + Duration::from_secs(7)));
    }

    #[test]
    fn the_daemon_wording_comes_from_n49_rather_than_from_here() {
        // If N49 renames the reason, this test moves with it instead of the screen drifting.
        assert_eq!(
            daemon_offline_label(),
            UnreachableReason::DaemonUnreachable
                .as_str()
                .replace('_', " ")
        );
    }
}
