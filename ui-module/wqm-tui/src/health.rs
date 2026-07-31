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
//! that returns **unhealthy** raises its own alarm, because that genuinely is news the
//! all-clear would otherwise hide. A component that returns healthy raises nothing — the
//! daemon's all-clear already said so.
//!
//! # The sustained half is not a toast
//!
//! A toast announces a transition and expires. "The daemon is unreachable" has to stay said
//! for as long as it is true, which is [`crate::tokens::Condition`] and the red wash
//! ([`crate::widgets::surface`]). [`SystemHealth::condition`] is where the one state becomes
//! the other, so the wash and the toast can never disagree about whether the daemon is up.

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

/// Every toast that a move from `before` to `after` is allowed to raise, in the order they
/// should be pushed.
///
/// This is the whole alert policy, and it is deliberately the only way toasts are produced
/// from health: a caller that pushed its own would be able to bypass the silencing rules.
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

    for component in &after.components {
        let previous = before.component(&component.role);

        let raise = if before.daemon == Health::Offline {
            // Coming back: an unhealthy component is news the daemon's all-clear would hide.
            // A healthy one is not — the all-clear already said it.
            component.health != Health::Healthy
        } else {
            match previous {
                // A component that was already known: only a real change speaks.
                Some(was) => was != component.health,
                // One that has just appeared: news only if it arrives in trouble.
                None => component.health != Health::Healthy,
            }
        };

        if !raise {
            continue;
        }

        let from = previous
            .filter(|_| before.daemon != Health::Offline)
            // A component with no previous reading — new, or seen through a daemon that was
            // down — is treated as having been healthy, so `transition` still describes a
            // change and cannot be handed the same state twice.
            .unwrap_or(Health::Healthy);
        let from = if from == component.health {
            Health::Healthy
        } else {
            from
        };

        toasts.extend(Toast::transition(
            from,
            component.health,
            format!("{} {}", component.role, verb(component.health)),
        ));
    }

    // A component that disappeared from the report says nothing: its absence is not a state,
    // and `store_health` already shows a vanished row as unreadable rather than dropping it.
    toasts
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
