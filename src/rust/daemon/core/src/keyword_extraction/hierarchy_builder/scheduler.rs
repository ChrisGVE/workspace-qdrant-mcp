//! Scheduling helpers for nightly hierarchy rebuild.

/// Calculate the delay until the next 2:00 AM local time.
pub(super) fn delay_until_next_2am() -> std::time::Duration {
    let now = chrono::Local::now();
    let today_2am = now
        .date_naive()
        .and_hms_opt(2, 0, 0)
        .expect("valid time");

    let target = if now.naive_local() < today_2am {
        // 2 AM hasn't passed today yet
        today_2am
    } else {
        // 2 AM already passed, schedule for tomorrow
        today_2am + chrono::Duration::days(1)
    };

    let target_local = target
        .and_local_timezone(now.timezone())
        .single()
        .unwrap_or_else(|| {
            // DST ambiguity fallback: use the latest option
            target
                .and_local_timezone(now.timezone())
                .latest()
                .expect("at least one valid local time")
        });

    let diff = target_local - now;
    diff.to_std().unwrap_or(std::time::Duration::from_secs(3600))
}
