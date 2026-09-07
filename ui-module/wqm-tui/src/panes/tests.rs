//! What the panes OUTSIDE the Dashboard are pinned to.
//!
//! One cross-pane invariant lives here rather than in either pane's own file, because it is a
//! statement about the pair: the Dashboard's new focus treatment (2026-09-07 — a selector block
//! on the heading, and the key letter dropped) must not reach the Service hub's zones, which
//! keep the `▌` bar. A guard inside `status_band` could only say "this pane still renders";
//! this one says "these panes render what they rendered before the Dashboard changed".

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

use crate::panes::status_band::StatusBand;
use crate::panes::storage::StorageCell;
use crate::tokens::Health;
use crate::views::service::frames;
use crate::widgets::chrome::test_support::Restore;
use crate::widgets::chrome::Attention;

/// A digest of every painted cell: its symbol, and the whole of its style.
///
/// FNV-1a, spelled out here rather than pulled in, so the number below is reproducible from
/// this file alone. It covers the STYLE as well as the glyph, which is the point — a heading
/// that had silently changed from `▌ Storage` in normal weight to an inverted block would draw
/// the same letters.
fn digest(buf: &Buffer) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            let cell = buf.cell((x, y)).expect("cell in area");
            for byte in format!("{}|{:?};", cell.symbol(), cell.style()).bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    hash
}

fn drawn(widget: impl Widget, width: u16, height: u16) -> Buffer {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    widget.render(area, &mut buf);
    buf
}

/// The Service hub's two panes render exactly what they rendered before the Dashboard's cells
/// took the selector block (2026-09-07).
///
/// The expected digests were taken from the commit BEFORE that change and written here, so this
/// is a comparison against a recorded past rather than against a second render of the present
/// code — which would agree with any drift at all, in both copies at once.
///
/// Chris has ruled the block treatment for the Dashboard's cells only. Whether it generalises
/// to every focused zone is still open; until it is answered, this guard is what stops the
/// answer being given by accident.
#[test]
fn the_service_hubs_panes_render_exactly_what_they_did_before_the_dashboard_took_the_block() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const STATUS_BAND_IDLE: u64 = 0xd134_d964_949a_18e2;
    const STATUS_BAND_FOCUSED: u64 = 0xfba3_9a37_6402_c66c;
    const STORAGE_IDLE: u64 = 0xcba2_d0ef_118e_bb6c;
    const STORAGE_FOCUSED: u64 = 0xdcbb_6fb3_65f2_4a1c;

    let band = |attention| {
        StatusBand::new(
            frames::stores(Health::Degraded),
            frames::serving(),
            0,
            attention,
        )
    };
    let storage = |attention| {
        StorageCell::new(
            frames::stores(Health::Degraded),
            frames::serving(),
            1,
            attention,
        )
    };

    for (label, actual, expected) in [
        (
            "status band, no zone focused",
            digest(&drawn(band(Attention::None), 80, StatusBand::ROWS)),
            STATUS_BAND_IDLE,
        ),
        (
            "status band, focused",
            digest(&drawn(band(Attention::Zone(0)), 80, StatusBand::ROWS)),
            STATUS_BAND_FOCUSED,
        ),
        (
            "storage, no zone focused",
            digest(&drawn(storage(Attention::None), 59, 8)),
            STORAGE_IDLE,
        ),
        (
            "storage, focused",
            digest(&drawn(storage(Attention::Zone(1)), 59, 8)),
            STORAGE_FOCUSED,
        ),
    ] {
        assert_eq!(
            actual, expected,
            "{label}: this pane's render changed — 0x{actual:016x}"
        );
    }
}
