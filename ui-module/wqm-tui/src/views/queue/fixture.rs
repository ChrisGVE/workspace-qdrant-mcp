//! The Queue tab's rows — two hundred of them, every one read off the running v0.1.
//!
//! **Real rows, captured, never invented** — the discipline
//! [`crate::views::dashboard::frames`] already follows, and the reason is the same: a fixture of
//! plausible file names looks fine and teaches nothing. `…a705d0205/tool-results/bot1i2cqf.txt`
//! is what a queued object actually looks like on this machine, and a forty-four-column `Object`
//! field surviving it is the fact the layout has to earn.
//!
//! # How they were gathered, and where the composition is NOT a single page
//!
//! `wqm tui` was driven under tmux at 125 × 34 (`tmux new-session -d -x 125 -y 34 'wqm tui'`,
//! then `2`, then `C-f` until the rows repeated), and again with its `s` key cycling the status
//! filter, so that every state the column can show has real rows behind it. That second half is
//! why this is a **composed** page rather than a transcript: v0.1's own buffer is the two
//! hundred NEWEST items, and on this workspace those are all `pending` — the ten `in progress`
//! rows are two hours old and the three `failed` ones nineteen. A fixture drawn from one page
//! would have shown a `Status` column of one word in one hue, which is a column that proves
//! nothing about the rule it exists to demonstrate. So the three status pages were merged and
//! re-ordered by age, and the two oldest `pending` rows dropped to land on exactly
//! [`crate::panes::list::LIST_PAGE`]. Nothing here was typed; what was chosen is which real rows
//! to keep.
//!
//! # What was dropped, and why the list is short by one
//!
//! One captured row carried a tenant (`local_bf`) that is **not** among the project names this
//! repository already publishes in [`crate::views::dashboard::frames`]. This crate is public, so
//! the vetted set is exactly the names already in that file — its seven `PROJECTS`, its two
//! `ACTIVE`, and the `PlotSwift` its `ERRORS` name — and every row outside it was dropped. One
//! row, silently, as the only thing a name-vetting rule can honestly do.
//!
//! # The `Object` strings are v0.1's own drawn text
//!
//! A row whose object begins with `…` is one **v0.1** had already shortened into its own
//! thirty-seven-column field; the full path was never on the screen to read. So the fixture is
//! honest about what was captured rather than reconstructing paths nobody saw — and this crate's
//! own left-elision is still exercised, because the `Small 80x24` frame gives the column far
//! fewer than thirty-seven columns.
//!
//! # `bytes` and `seconds` are ORDERS, not measurements
//!
//! `30.8 KB` is what v0.1 drew; the true byte count is not recoverable from it. What is
//! recoverable — and all a sort needs — is the ordering, which is monotone in the printed value.
//! So `bytes` is the displayed figure scaled by its unit (1 KB = 1024 B) and `seconds` the
//! displayed age in seconds: two numbers that order the column exactly as a reader would expect,
//! and that claim nothing more precise than the screen they came from.
//!
//! **NOT contract-bound (UIQ-013 style).** No wire message carries these fields. `wqm-common`
//! names the collections and the response envelope; it names none of this. The shape is v0.1's
//! captured screen, and it is labelled a fiction rather than quietly treated as one.

use super::state::Kind;
use super::state::Status::{self, Failed, InProgress, Pending};

/// One queued item, as the Queue tab shows it. **NOT contract-bound (UIQ pending).**
pub struct QueueRow {
    /// The row's reference number — 1-based, assigned at load, and **invariant** under every
    /// sort, filter and selector this screen has. It is a field rather than the drawn index
    /// precisely so that "row 137" still means row 137 after the list has been reordered.
    pub no: u16,
    /// What the one-letter `T` column shows. Every captured row is a project item.
    pub kind: Kind,
    pub tenant: &'static str,
    /// v0.1's own drawn text — already shortened by v0.1 where it begins with `…`.
    pub object: &'static str,
    /// The item type the `Type` column shows. Every captured row is a `file`; the field is
    /// per-row anyway, because a column whose value were a constant would be a lie the first
    /// time it was not.
    pub item: &'static str,
    pub op: &'static str,
    pub status: Status,
    /// What v0.1 drew — empty on a delete, which has no size.
    pub size: &'static str,
    /// What that figure ORDERS as. See the module docs: an order, not a measurement.
    pub bytes: u64,
    pub age: &'static str,
    pub seconds: u64,
}

/// Every captured row is a project item (`P`) holding a `file`, so this constructor states both
/// once instead of two hundred rows repeating them. They are still per-row FIELDS: a library or
/// a non-file item is written as a struct literal and says so for itself, which a column read
/// from a constant could never do.
///
/// `size` and `age` arrive as pairs because that is what they are — a printed form and the
/// magnitude it orders by, which is the whole content of [`crate::panes::cell::Cell::Measured`].
/// Splitting them into four arguments would let a row be written with its bytes beside the
/// wrong units, and would also be four arguments nobody could keep in order.
const fn row(
    no: u16,
    tenant: &'static str,
    object: &'static str,
    op: &'static str,
    status: Status,
    size: (&'static str, u64),
    age: (&'static str, u64),
) -> QueueRow {
    QueueRow {
        no,
        kind: Kind::Project,
        tenant,
        object,
        item: "file",
        op,
        status,
        size: size.0,
        bytes: size.1,
        age: age.0,
        seconds: age.1,
    }
}

/// The captured page, newest first — the order v0.1's own list is in.
///
/// `static` rather than `const`: two hundred rows is a table to point at, not a value to copy in
/// at every use. `#[rustfmt::skip]` because this is a **table**, and the one property that makes
/// two hundred rows readable is that each is one line with its fields in the same columns.
/// Formatted at the crate's width every row becomes a dozen lines and the table becomes
/// twenty-four hundred lines of vertical prose that no reader can compare down a column.
#[rustfmt::skip]
pub static ROWS: [QueueRow; 200] = [
    row(  1, "claude", "…a705d0205/tool-results/bapnce3uu.txt", "update", Pending, ("30.8 KB", 31539), ("1m ago", 60)),
    row(  2, ".config", "…a705d0205/tool-results/bot1i2cqf.txt", "update", Pending, ("30.8 KB", 31539), ("1m ago", 60)),
    row(  3, "claude", "…ts/agent-aa7d7964330b31483.meta.json", "update", Pending, ("169 B", 169), ("2m ago", 120)),
    row(  4, ".config", "herdr/sessions/Main/session.json", "update", Pending, ("10.6 KB", 10854), ("4m ago", 240)),
    row(  5, "claude", "…ts/agent-a1eff1edcbad2c30e.meta.json", "update", Pending, ("319 B", 319), ("6m ago", 360)),
    row(  6, "open-books", "…uilding/common/filters/rust-xref.lua", "update", Pending, ("", 0), ("7m ago", 420)),
    row(  7, "open-books", "…ommon/filters/test_collapse_xref.lua", "update", Pending, ("", 0), ("7m ago", 420)),
    row(  8, "workspace-qdrant-mcp", "…-ux/GT001-wqm-tui/VISUAL-LANGUAGE.md", "update", Pending, ("39.8 KB", 40755), ("7m ago", 420)),
    row(  9, "open-books", "…-workspace/s334_single_band_probe.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 10, "open-books", "…ding/common/stage_b/reading_guide.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 11, "open-books", "…ommon/stage_b/reading_guide_bands.py", "add", Pending, ("", 0), ("7m ago", 420)),
    row( 12, "open-books", "…de-variants/reading-guide-round3.pdf", "add", Pending, ("", 0), ("7m ago", 420)),
    row( 13, "open-books", "…ommon/stage_b/reading_guide_latex.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 14, "open-books", "output/global.toml", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 15, "open-books", "…e/reading-guide-variants/VARIANTS.md", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 16, "open-books", "…ing/common/filters/collapse_xref.lua", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 17, "open-books", "CLAUDE.md", "update", Pending, ("24.6 KB", 25190), ("7m ago", 420)),
    row( 18, "open-books", "…ixtures/preamble/golden/expected.tex", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 19, "open-books", "…lding/common/tools/mkreadingguide.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 20, "open-books", "…ding/common/filters/xref_locator.lua", "add", Pending, ("", 0), ("7m ago", 420)),
    row( 21, "open-books", "…ng/common/filters/test_rust_xref.lua", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 22, "open-books", "output/level.schema.json", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 23, "open-books", "…common/stage_b/reading_guide_scan.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 24, "open-books", "output/global.schema.json", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 25, "open-books", "…g/tests/test_reading_guide_config.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 26, "open-books", "…ols/tests/test_collapse_band_span.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 27, "open-books", "…/stage_b/tests/test_reading_guide.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 28, "open-books", "…_building/common/preamble/header.tex", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 29, "open-books", "…g/tests/test_collapse_band_single.py", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 30, "open-books", "…ATTING_RULES/05-url-link-handling.md", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 31, "open-books", "…es/reading-guide/golden/expected.tex", "update", Pending, ("", 0), ("7m ago", 420)),
    row( 32, "workspace-qdrant-mcp", "…le/wqm-tui/src/widgets/chrome/mod.rs", "update", Pending, ("", 0), ("8m ago", 480)),
    row( 33, "workspace-qdrant-mcp", "…odule/wqm-tui/src/panes/cell/sort.rs", "add", Pending, ("", 0), ("8m ago", 480)),
    row( 34, "workspace-qdrant-mcp", "…ui/src/views/dashboard/tests/sort.rs", "add", Pending, ("", 0), ("8m ago", 480)),
    row( 35, "workspace-qdrant-mcp", "…ui/src/views/dashboard/ingredient.rs", "update", Pending, ("", 0), ("8m ago", 480)),
    row( 36, "workspace-qdrant-mcp", "…/wqm-tui/src/widgets/chrome/keyed.rs", "add", Pending, ("", 0), ("8m ago", 480)),
    row( 37, ".config", "figlet/uskata.flc", "add", Pending, ("747 B", 747), ("21m ago", 1260)),
    row( 38, ".config", "…ts/default/nextcloud_log.json.sample", "add", Pending, ("3.3 KB", 3379), ("21m ago", 1260)),
    row( 39, ".config", "…89401/tools/get-svg-component.js.map", "add", Pending, ("461.3 KB", 472371), ("21m ago", 1260)),
    row( 40, ".config", "…ant/qdrant-web-ui/public/logo192.png", "add", Pending, ("6.0 KB", 6144), ("21m ago", 1260)),
    row( 41, ".config", "…30581ac/toggle-minimal-effect.js.map", "add", Pending, ("2.5 KB", 2560), ("21m ago", 1260)),
    row( 42, ".config", "…-30f96a3c9f83/cmd-show-status.js.map", "add", Pending, ("349.6 KB", 357990), ("21m ago", 1260)),
    row( 43, ".config", "figlet/5lineoblique.flf", "add", Pending, ("8.6 KB", 8806), ("21m ago", 1260)),
    row( 44, ".config", "…42f1-baa9-1541966a1e00/custom.js.map", "add", Pending, ("4.3 MB", 4508877), ("21m ago", 1260)),
    row( 45, ".config", "xm1/texmakerapp.ini", "add", Pending, ("25 B", 25), ("21m ago", 1260)),
    row( 46, ".config", "…ormats/default/uwsgi_log.json.sample", "add", Pending, ("3.6 KB", 3686), ("21m ago", 1260)),
    row( 47, ".config", "…5bd105e/tools/run-applescript.js.map", "add", Pending, ("1.4 KB", 1434), ("21m ago", 1260)),
    row( 48, ".config", "…memory/MEMORY.md.bak-20260825-130321", "add", Pending, ("2.0 KB", 2048), ("21m ago", 1260)),
    row( 49, ".config", "…d2570668dd6/search-userstyles.js.map", "add", Pending, ("876.7 KB", 897741), ("21m ago", 1260)),
    row( 50, ".config", "…addfadfbaf/assets/osicons/win-11.png", "add", Pending, ("195 B", 195), ("21m ago", 1260)),
    row( 51, ".config", "figlet/Bright.flf", "add", Pending, ("6.8 KB", 6963), ("21m ago", 1260)),
    row( 52, ".config", "epy/states.db", "add", Pending, ("28.0 KB", 28672), ("21m ago", 1260)),
    row( 53, ".config", "figlet/SL Script.flf", "add", Pending, ("4.8 KB", 4915), ("21m ago", 1260)),
    row( 54, ".config", "…f4-8a1520c55898/search-photos.js.map", "add", Pending, ("1.1 MB", 1153434), ("21m ago", 1260)),
    row( 55, ".config", "…15-b14c-fcb7ed51dbae/timeline.js.map", "add", Pending, ("3.1 MB", 3250586), ("21m ago", 1260)),
    row( 56, ".config", "…-8c8d4008c461/markdownpreview.js.map", "add", Pending, ("1.9 KB", 1946), ("21m ago", 1260)),
    row( 57, ".config", "…d72b8e/assets/screen-mirror-icon.png", "add", Pending, ("10.5 KB", 10752), ("21m ago", 1260)),
    row( 58, ".config", "…-481e-8bb9-44c4c6df53be/index.js.map", "add", Pending, ("66.9 KB", 68506), ("21m ago", 1260)),
    row( 59, ".config", "omp/agent/terminal-sessions/ttys003", "add", Pending, ("135 B", 135), ("21m ago", 1260)),
    row( 60, ".config", "figlet/henry3d.flf", "add", Pending, ("9.5 KB", 9728), ("21m ago", 1260)),
    row( 61, ".config", "weechat/xfer.conf", "add", Pending, ("981 B", 981), ("21m ago", 1260)),
    row( 62, ".config", "…95cf92e05cf6/assets/grab-delayed.png", "add", Pending, ("151.7 KB", 155341), ("21m ago", 1260)),
    row( 63, ".config", "figlet/Rozzo.flf", "add", Pending, ("7.5 KB", 7680), ("21m ago", 1260)),
    row( 64, ".config", "…d5ba06bb/assets/timezones/UTC-12.png", "add", Pending, ("39.9 KB", 40858), ("21m ago", 1260)),
    row( 65, ".config", "…/assets/compiled_raycast_swift/swift", "add", Pending, ("579.2 KB", 593101), ("21m ago", 1260)),
    row( 66, ".config", "lazygit/catppuccin/flake.lock", "add", Pending, ("2.4 KB", 2458), ("21m ago", 1260)),
    row( 67, ".config", "…/common/assets/git/leak-denylist.mcp", "add", Pending, ("964 B", 964), ("21m ago", 1260)),
    row( 68, ".config", "herdr/config.toml", "update", Pending, ("27.3 KB", 27955), ("21m ago", 1260)),
    row( 69, ".config", "…3c87730581ac/toggle-auto-hide.js.map", "add", Pending, ("3.0 KB", 3072), ("21m ago", 1260)),
    row( 70, ".config", "…c2f583/assets/codesandbox-bright.png", "add", Pending, ("671 B", 671), ("21m ago", 1260)),
    row( 71, ".config", "…nks/tmp-1mmfhhwpdy6hi/mnemopi.db-wal", "add", Pending, ("2.3 MB", 2411725), ("21m ago", 1260)),
    row( 72, "open-books", "…ools/tests/test_collapse_codesize.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row( 73, ".config", "…0-44b7-9446-55b52a1cac36/open.js.map", "add", Pending, ("996.8 KB", 1020723), ("21m ago", 1260)),
    row( 74, ".config", "figlet/Graffiti.flf", "add", Pending, ("6.4 KB", 6554), ("21m ago", 1260)),
    row( 75, ".config", "…/bat/themes/Catppuccin Latte.tmTheme", "add", Pending, ("62.8 KB", 64307), ("21m ago", 1260)),
    row( 76, ".config", "…3-c34feb6486ee/assets/arxiv-icon.png", "add", Pending, ("30.2 KB", 30925), ("21m ago", 1260)),
    row( 77, "workspace-qdrant-mcp", "…/wqm-tui/src/widgets/chrome/keyed.rs", "update", Pending, ("4.8 KB", 4915), ("21m ago", 1260)),
    row( 78, "open-books", "…ace/s335-replay/plan-s335-after.json", "add", Pending, ("", 0), ("21m ago", 1260)),
    row( 79, ".config", "…09c22a6dd8/create-docset-link.js.map", "add", Pending, ("5.5 KB", 5632), ("21m ago", 1260)),
    row( 80, ".config", "…a4c6/tools/list-workflow-runs.js.map", "add", Pending, ("2.9 MB", 3040870), ("21m ago", 1260)),
    row( 81, "open-books", "…lding/common/tools/collapse_align.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row( 82, ".config", "snyk/ls-config-Neovim", "add", Pending, ("4.0 KB", 4096), ("21m ago", 1260)),
    row( 83, ".config", "…-4d34ead78777/assets/branch@dark.svg", "add", Pending, ("339 B", 339), ("21m ago", 1260)),
    row( 84, ".config", "…kills/tui-test-harness/bin/tui-close", "add", Pending, ("1.4 KB", 1434), ("21m ago", 1260)),
    row( 85, "open-books", "…uilding/common/tools/collapse_lex.py", "add", Pending, ("", 0), ("21m ago", 1260)),
    row( 86, ".config", "…64c-4dba-ba77-b1cb543256d6/ai.js.map", "add", Pending, ("272.2 KB", 278733), ("21m ago", 1260)),
    row( 87, ".config", "…de-616a22387eae/assets/icon-stop.png", "add", Pending, ("231 B", 231), ("21m ago", 1260)),
    row( 88, ".config", "…d5ba06bb/assets/timezones/UTC+10.png", "add", Pending, ("41.4 KB", 42394), ("21m ago", 1260)),
    row( 89, ".config", "…nsupported-simulators.command.js.map", "add", Pending, ("9.6 KB", 9830), ("21m ago", 1260)),
    row( 90, ".config", "…46b-a8f9-d710b41aa36d/grammar.js.map", "add", Pending, ("3.2 MB", 3355443), ("21m ago", 1260)),
    row( 91, ".config", "…a-a5c1-1c26f5b329a9/assets/issue.png", "add", Pending, ("1.1 KB", 1126), ("21m ago", 1260)),
    row( 92, ".config", "…-4750-8b4a-81e748f2c33a/index.js.map", "add", Pending, ("309.2 KB", 316621), ("21m ago", 1260)),
    row( 93, ".config", "…12a465/fan-add-event-editable.js.map", "add", Pending, ("1.1 MB", 1153434), ("21m ago", 1260)),
    row( 94, ".config", "figlet/Isometric1.flf", "add", Pending, ("12.5 KB", 12800), ("21m ago", 1260)),
    row( 95, ".config", "Code/TransportSecurity", "add", Pending, ("5.3 KB", 5427), ("21m ago", 1260)),
    row( 96, ".config", "…ts/prompts-library-db.0.mdb/data.mdb", "add", Pending, ("12.0 KB", 12288), ("21m ago", 1260)),
    row( 97, ".config", "…b-4c8d-9bfd-8c8d4008c461/auto.js.map", "add", Pending, ("1.8 KB", 1843), ("21m ago", 1260)),
    row( 98, ".config", "…-95bf-4d34ead78777/assets/branch.svg", "add", Pending, ("339 B", 339), ("21m ago", 1260)),
    row( 99, ".config", "…37fc20bd0c2/formatToJsonValue.js.map", "add", Pending, ("1.1 MB", 1153434), ("21m ago", 1260)),
    row(100, "open-books", "…workspace/s335-replay/plan-s335.json", "add", Pending, ("", 0), ("21m ago", 1260)),
    row(101, "open-books", "…ools/tests/test_collapse_identity.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row(102, "workspace-qdrant-mcp", "…odule/wqm-tui/src/panes/cell/sort.rs", "update", Pending, ("2.4 KB", 2458), ("21m ago", 1260)),
    row(103, "open-books", "…ng/common/tools/collapse_identity.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row(104, "open-books", "…tools/tests/test_collapse_aligned.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row(105, ".config", "…-be8a-f740b3f1656a/assets/remote.svg", "add", Pending, ("413 B", 413), ("21m ago", 1260)),
    row(106, ".config", "…8857d5ba06bb/assets/command-icon.png", "add", Pending, ("91.8 KB", 94003), ("21m ago", 1260)),
    row(107, ".config", "…/toggle-stickies-float-on-top.js.map", "add", Pending, ("1.1 MB", 1153434), ("21m ago", 1260)),
    row(108, ".config", "…46-55b52a1cac36/assets/list-icon.png", "add", Pending, ("5.4 KB", 5530), ("21m ago", 1260)),
    row(109, ".config", "…-87ccac1299a1/tools/open-path.js.map", "add", Pending, ("795 B", 795), ("21m ago", 1260)),
    row(110, "open-books", "…n/tools/tests/test_collapse_delta.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row(111, ".config", "eza/eza-themes/test_dir/song.flac", "add", Pending, ("0 B", 0), ("21m ago", 1260)),
    row(112, ".config", "…-4724-afb9-9835e315ba92/index.js.map", "add", Pending, ("526.0 KB", 538624), ("21m ago", 1260)),
    row(113, ".config", "…-616a22387eae/assets/docker-icon.png", "add", Pending, ("64.6 KB", 66150), ("21m ago", 1260)),
    row(114, ".config", "figlet/JS Cursive.flf", "add", Pending, ("3.5 KB", 3584), ("21m ago", 1260)),
    row(115, ".config", "figlet/miniwi.flf", "add", Pending, ("2.7 KB", 2765), ("21m ago", 1260)),
    row(116, ".config", "…60725f2a39/assets/status-loading.png", "add", Pending, ("351 B", 351), ("21m ago", 1260)),
    row(117, ".config", "…ormats/default/caddy_log.json.sample", "add", Pending, ("3.6 KB", 3686), ("21m ago", 1260)),
    row(118, ".config", "…583db3f4f/word-synonym-search.js.map", "add", Pending, ("269.3 KB", 275763), ("21m ago", 1260)),
    row(119, ".config", "codex/common/assets/git/leak-denylist", "add", Pending, ("2.8 KB", 2867), ("21m ago", 1260)),
    row(120, ".config", "…r/sessions/Main/session-history.json", "update", Pending, ("576.2 KB", 590029), ("21m ago", 1260)),
    row(121, ".config", "…za-themes/imgs/catppuccin-frappe.png", "add", Pending, ("186.9 KB", 191386), ("21m ago", 1260)),
    row(122, ".config", "…onfigs/default/uk-keymap.json.sample", "add", Pending, ("386 B", 386), ("21m ago", 1260)),
    row(123, "open-books", "…n/tools/tests/test_collapse_alias.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row(124, ".config", "…s/default/lnav_debug_log.json.sample", "add", Pending, ("3.6 KB", 3686), ("21m ago", 1260)),
    row(125, ".config", "…efault/github_events_log.json.sample", "add", Pending, ("5.8 KB", 5939), ("21m ago", 1260)),
    row(126, "open-books", "…e/s335-replay/plan-s335-arm-tok.json", "add", Pending, ("", 0), ("21m ago", 1260)),
    row(127, ".config", "…s/btop/themes/catppuccin_mocha.theme", "add", Pending, ("2.2 KB", 2253), ("21m ago", 1260)),
    row(128, ".config", "…cb543256d6/tools/list-folders.js.map", "add", Pending, ("247.8 KB", 253747), ("21m ago", 1260)),
    row(129, ".config", "…ed4e9d8ed1/assets/microsoftAzure.png", "add", Pending, ("70.4 KB", 72090), ("21m ago", 1260)),
    row(130, "open-books", "…lding/common/tools/collapse_alias.py", "update", Pending, ("", 0), ("21m ago", 1260)),
    row(131, ".config", "…default/lnav-breakpoint-handler.lnav", "add", Pending, ("428 B", 428), ("21m ago", 1260)),
    row(132, ".config", "figlet/Broadway.flf", "add", Pending, ("14.7 KB", 15053), ("21m ago", 1260)),
    row(133, ".config", "figlet/Shadow.flf", "add", Pending, ("14.2 KB", 14541), ("21m ago", 1260)),
    row(134, ".config", "…5d-bd84-eb02c43eb904/releases.js.map", "add", Pending, ("723.7 KB", 741069), ("21m ago", 1260)),
    row(135, ".config", "…ugins/config/herdr-lazy/plugins.list", "add", Pending, ("577 B", 577), ("21m ago", 1260)),
    row(136, ".config", "…ts/default/procstate_log.json.sample", "add", Pending, ("974 B", 974), ("21m ago", 1260)),
    row(137, ".config", "…62-95cf92e05cf6/assets/load-file.png", "add", Pending, ("116.0 KB", 118784), ("21m ago", 1260)),
    row(138, ".config", "…a6-fca9-7000-b758-d7179632b993.jsonl", "add", Pending, ("4.2 KB", 4301), ("22m ago", 1320)),
    row(139, ".config", "…sets/compiled_raycast_swift/contacts", "add", Pending, ("674.6 KB", 690790), ("22m ago", 1320)),
    row(140, ".config", "…12a465/fan-quick-add-reminder.js.map", "add", Pending, ("1.1 MB", 1153434), ("22m ago", 1320)),
    row(141, ".config", "…a7-da8b-7000-89e7-4655e02d9e03.jsonl", "add", Pending, ("4.1 KB", 4198), ("22m ago", 1320)),
    row(142, ".config", "figlet/646-cn.flc", "add", Pending, ("4.9 KB", 5018), ("22m ago", 1320)),
    row(143, ".config", "…-9bfd-8c8d4008c461/colortools.js.map", "add", Pending, ("1.8 KB", 1843), ("22m ago", 1320)),
    row(144, ".config", "…daddfadfbaf/assets/osicons/linux.png", "add", Pending, ("814 B", 814), ("22m ago", 1320)),
    row(145, ".config", "figlet/Script.flf", "add", Pending, ("16.5 KB", 16896), ("22m ago", 1320)),
    row(146, ".config", "figlet/Double Shorts.flf", "add", Pending, ("3.4 KB", 3482), ("22m ago", 1320)),
    row(147, ".config", "…465/fan-add-reminder-editable.js.map", "add", Pending, ("1.1 MB", 1153434), ("22m ago", 1320)),
    row(148, ".config", "eza/eza-themes/test_dir/file.mp4", "add", Pending, ("0 B", 0), ("22m ago", 1320)),
    row(149, ".config", "lnav/cmd.history", "add", Pending, ("0 B", 0), ("22m ago", 1320)),
    row(150, ".config", "…6b-a8f9-d710b41aa36d/friendly.js.map", "add", Pending, ("3.2 MB", 3355443), ("22m ago", 1320)),
    row(151, ".config", "…open-books/memory/MEMORY.md.bak-s106", "add", Pending, ("15.0 KB", 15360), ("22m ago", 1320)),
    row(152, ".config", "…formats/default/page_log.json.sample", "add", Pending, ("2.4 KB", 2458), ("22m ago", 1320)),
    row(153, ".config", "figlet/Mini.flf", "add", Pending, ("8.9 KB", 9114), ("22m ago", 1320)),
    row(154, ".config", "figlet/646-es2.flc", "add", Pending, ("4.9 KB", 5018), ("22m ago", 1320)),
    row(155, ".config", "…-9f40-75fcb039a4c6/assets/vscode.svg", "add", Pending, ("483 B", 483), ("22m ago", 1320)),
    row(156, ".config", "…e8-4373-8162-95cf92e05cf6/ssw.js.map", "add", Pending, ("2.8 KB", 2867), ("22m ago", 1320)),
    row(157, ".config", "figlet/amcthin.flf", "add", Pending, ("6.9 KB", 7066), ("22m ago", 1320)),
    row(158, ".config", "…tools/xcode-add-swift-package.js.map", "add", Pending, ("450.7 KB", 461517), ("22m ago", 1320)),
    row(159, ".config", "…96a3c9f83/cmd-clean-snapshots.js.map", "add", Pending, ("32.1 KB", 32870), ("22m ago", 1320)),
    row(160, ".config", "…8162-95cf92e05cf6/assets/uploads.png", "add", Pending, ("137.6 KB", 140902), ("22m ago", 1320)),
    row(161, ".config", "…-459a-a5c1-1c26f5b329a9/index.js.map", "add", Pending, ("2.6 MB", 2726298), ("22m ago", 1320)),
    row(162, ".config", "…f-885c-acf4ebcf19b1/summarize.js.map", "add", Pending, ("4.8 MB", 5033165), ("22m ago", 1320)),
    row(163, ".config", "figlet/koi8r.flc", "add", Pending, ("2.0 KB", 2048), ("22m ago", 1320)),
    row(164, ".config", "…c6f6c93/paste-latest-otp-code.js.map", "add", Pending, ("268.2 KB", 274637), ("22m ago", 1320)),
    row(165, ".config", "zsh/themes/bat/assets/macchiato.webp", "add", Pending, ("62.7 KB", 64205), ("22m ago", 1320)),
    row(166, ".config", "…96de586cef/see-important-mail.js.map", "add", Pending, ("8.6 MB", 9017754), ("22m ago", 1320)),
    row(167, ".config", "…507-665fd0e11784/conversation.js.map", "add", Pending, ("1.7 MB", 1782579), ("22m ago", 1320)),
    row(168, ".config", "…6-8b11-261e5c6f6c93/open-chat.js.map", "add", Pending, ("486.2 KB", 497869), ("22m ago", 1320)),
    row(169, ".config", "…/default/zap_console_log.json.sample", "add", Pending, ("1.4 KB", 1434), ("22m ago", 1320)),
    row(170, ".config", "…d-9bfd-8c8d4008c461/yaml2json.js.map", "add", Pending, ("1.8 KB", 1843), ("22m ago", 1320)),
    row(171, ".config", "…84-eb02c43eb904/assets/icon@dark.png", "add", Pending, ("153.6 KB", 157286), ("22m ago", 1320)),
    row(172, ".config", "…9f40-75fcb039a4c6/my-projects.js.map", "add", Pending, ("3.1 MB", 3250586), ("22m ago", 1320)),
    row(173, ".config", "…225debe4eeb85218468e5f6fd6466f572013", "add", Pending, ("67 B", 67), ("22m ago", 1320)),
    row(174, ".config", "figlet/Doom.flf", "add", Pending, ("7.4 KB", 7578), ("22m ago", 1320)),
    row(175, ".config", "…d48a98/assets/1password-settings.png", "add", Pending, ("238.4 KB", 244122), ("22m ago", 1320)),
    row(176, ".config", "…-unsupported-runtimes.command.js.map", "add", Pending, ("9.8 KB", 10035), ("22m ago", 1320)),
    row(177, ".config", "figlet/Efti Wall.flf", "add", Pending, ("7.1 KB", 7270), ("22m ago", 1320)),
    row(178, ".config", "…/create-swift-package.command.js.map", "add", Pending, ("1.5 MB", 1572864), ("22m ago", 1320)),
    row(179, ".config", "…git/catppuccin/assets/macchiato.webp", "add", Pending, ("47.3 KB", 48435), ("22m ago", 1320)),
    row(180, ".config", "figlet/Computer.flf", "add", Pending, ("6.5 KB", 6656), ("22m ago", 1320)),
    row(181, ".config", "…-211216310ee5/recentDownloads.js.map", "add", Pending, ("816.8 KB", 836403), ("22m ago", 1320)),
    row(182, ".config", "…s/default/default-keymap.json.sample", "add", Pending, ("9.4 KB", 9626), ("22m ago", 1320)),
    row(183, ".config", "…ills/tui-test-harness/bin/tui-launch", "add", Pending, ("5.8 KB", 5939), ("22m ago", 1320)),
    row(184, ".config", "figlet/Lean.flf", "add", Pending, ("27.9 KB", 28570), ("22m ago", 1320)),
    row(185, ".config", "…/search-code-snippets.command.js.map", "add", Pending, ("1.5 MB", 1572864), ("22m ago", 1320)),
    row(186, ".config", "…-4851-a6d3-c34feb6486ee/index.js.map", "add", Pending, ("2.7 MB", 2831155), ("22m ago", 1320)),
    row(187, ".config", "codex/common/scripts/mcp/context7", "add", Pending, ("136 B", 136), ("22m ago", 1320)),
    row(188, "workspace-qdrant-mcp", "…ves/prd-workspace/audit_report_r1.md", "add", InProgress, ("23.2 KB", 23757), ("2h ago", 7200)),
    row(189, "workspace-qdrant-mcp", "…ves/prd-workspace/audit_report_r5.md", "add", InProgress, ("12.6 KB", 12902), ("2h ago", 7200)),
    row(190, "workspace-qdrant-mcp", "…-workspace/audit-r5-consolidation.md", "add", InProgress, ("14.8 KB", 15155), ("2h ago", 7200)),
    row(191, "workspace-qdrant-mcp", "…s/prd-workspace/audit-r3-security.md", "add", InProgress, ("8.0 KB", 8192), ("2h ago", 7200)),
    row(192, "workspace-qdrant-mcp", "…workspace/audit-r3-implementation.md", "add", InProgress, ("13.9 KB", 14234), ("2h ago", 7200)),
    row(193, "workspace-qdrant-mcp", "…rd-workspace/designer-response-r7.md", "add", InProgress, ("10.6 KB", 10854), ("2h ago", 7200)),
    row(194, "workspace-qdrant-mcp", "…ves/prd-workspace/audit-r4-domain.md", "add", InProgress, ("10.6 KB", 10854), ("2h ago", 7200)),
    row(195, "workspace-qdrant-mcp", "…rchives/prd-workspace/audit-r4-ux.md", "add", InProgress, ("7.5 KB", 7680), ("2h ago", 7200)),
    row(196, "workspace-qdrant-mcp", "…ves/prd-workspace/audit_report_r2.md", "add", InProgress, ("20.8 KB", 21299), ("2h ago", 7200)),
    row(197, "workspace-qdrant-mcp", "…s/prd-workspace/audit-r4-security.md", "add", InProgress, ("7.9 KB", 8090), ("2h ago", 7200)),
    row(198, "PlotSwift", "…ces/PlotSwift/Axes+Decorations.swift", "add", Failed, ("8.3 KB", 8499), ("19h ago", 68400)),
    row(199, "PlotSwift", "…lotSwiftTests/AnnotationsTests.swift", "add", Failed, ("4.3 KB", 4403), ("19h ago", 68400)),
    row(200, "PlotSwift", "…otSwiftTests/SeabornPlotsTests.swift", "add", Failed, ("10.4 KB", 10650), ("19h ago", 68400)),
];
