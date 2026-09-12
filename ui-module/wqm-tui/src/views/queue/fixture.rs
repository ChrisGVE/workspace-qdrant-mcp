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
//! **NOT contract-bound (UIQ pending)** — the mark `views::dashboard::frames` carries, on
//! every one of its six shapes, and for the same reason. No wire message carries these fields. `wqm-common`
//! names the collections and the response envelope; it names none of this. The shape is v0.1's
//! captured screen, and it is labelled a fiction rather than quietly treated as one.

use super::state::{Kind, Op};
use super::state::Status::{self, Failed, InProgress, Pending};

/// One queued item, as the Queue tab shows it. **NOT contract-bound (UIQ pending).**
///
/// The row has no reference number of its own: the `No` column is the row's POSITION in the
/// list as displayed, computed by the list at render time, so a sort or a filter renumbers it
/// and no field here could survive one.
pub struct QueueRow {
    /// What the one-letter `T` column shows. Every captured row is a project item.
    pub kind: Kind,
    pub tenant: &'static str,
    /// v0.1's own drawn text — already shortened by v0.1 where it begins with `…`.
    pub object: &'static str,
    /// The item type the `Type` column shows. Every captured row is a `file`; the field is
    /// per-row anyway, because a column whose value were a constant would be a lie the first
    /// time it was not.
    pub item: &'static str,
    /// The operation the queue will perform on the item. Every captured row is an `update` or
    /// an `add` — no delete or scan was in flight when the page was taken, and the `o`
    /// selector's skip rule steps over the two the buffer holds none of.
    pub op: Op,
    pub status: Status,
    /// How large the item is, in bytes — **`None` when the size is not yet known**, which is a
    /// different fact from a size of zero and draws differently
    /// ([`crate::format::size`]). The figure is DERIVED from this at render (20260912,
    /// ruling 11): the row carries the magnitude, and how a magnitude is written down is the
    /// surface's rule rather than two hundred hand-typed strings that can disagree with it.
    pub bytes: Option<u64>,
    /// How long the item has been queued, in seconds. Derived at render, same reason.
    pub seconds: u64,
}

/// Every captured row is a project item (`P`) holding a `file`, so this constructor states both
/// once instead of two hundred rows repeating them. They are still per-row FIELDS: a library or
/// a non-file item is written as a struct literal and says so for itself, which a column read
/// from a constant could never do.
///
/// `bytes` and `seconds` are MAGNITUDES and nothing else (20260912, ruling 11). They used to
/// arrive as pairs — a printed form beside the number it orders by — and a pair is two chances
/// to write a row whose text and magnitude disagree, in a table of two hundred rows where
/// nobody would ever read both halves of the same line. The printed form is now produced from
/// the magnitude at render ([`crate::format::size`], [`crate::format::age`]), so the two cannot
/// drift and the surface's figure rules have one place to live.
const fn row(
    tenant: &'static str,
    object: &'static str,
    op: Op,
    status: Status,
    bytes: Option<u64>,
    seconds: u64,
) -> QueueRow {
    QueueRow {
        kind: Kind::Project,
        tenant,
        object,
        item: "file",
        op,
        status,
        bytes,
        seconds,
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
    row("claude", "…a705d0205/tool-results/bapnce3uu.txt", Op::Update, Pending, Some(31539), 60),
    row(".config", "…a705d0205/tool-results/bot1i2cqf.txt", Op::Update, Pending, Some(31539), 60),
    row("claude", "…ts/agent-aa7d7964330b31483.meta.json", Op::Update, Pending, Some(169), 120),
    row(".config", "herdr/sessions/Main/session.json", Op::Update, Pending, Some(10854), 240),
    row("claude", "…ts/agent-a1eff1edcbad2c30e.meta.json", Op::Update, Pending, Some(319), 360),
    row("open-books", "…uilding/common/filters/rust-xref.lua", Op::Update, Pending, None, 420),
    row("open-books", "…ommon/filters/test_collapse_xref.lua", Op::Update, Pending, None, 420),
    row("workspace-qdrant-mcp", "…-ux/GT001-wqm-tui/VISUAL-LANGUAGE.md", Op::Update, Pending, Some(40755), 420),
    row("open-books", "…-workspace/s334_single_band_probe.py", Op::Update, Pending, None, 420),
    row("open-books", "…ding/common/stage_b/reading_guide.py", Op::Update, Pending, None, 420),
    row("open-books", "…ommon/stage_b/reading_guide_bands.py", Op::Add, Pending, None, 420),
    row("open-books", "…de-variants/reading-guide-round3.pdf", Op::Add, Pending, None, 420),
    row("open-books", "…ommon/stage_b/reading_guide_latex.py", Op::Update, Pending, None, 420),
    row("open-books", "output/global.toml", Op::Update, Pending, None, 420),
    row("open-books", "…e/reading-guide-variants/VARIANTS.md", Op::Update, Pending, None, 420),
    row("open-books", "…ing/common/filters/collapse_xref.lua", Op::Update, Pending, None, 420),
    row("open-books", "CLAUDE.md", Op::Update, Pending, Some(25190), 420),
    row("open-books", "…ixtures/preamble/golden/expected.tex", Op::Update, Pending, None, 420),
    row("open-books", "…lding/common/tools/mkreadingguide.py", Op::Update, Pending, None, 420),
    row("open-books", "…ding/common/filters/xref_locator.lua", Op::Add, Pending, None, 420),
    row("open-books", "…ng/common/filters/test_rust_xref.lua", Op::Update, Pending, None, 420),
    row("open-books", "output/level.schema.json", Op::Update, Pending, None, 420),
    row("open-books", "…common/stage_b/reading_guide_scan.py", Op::Update, Pending, None, 420),
    row("open-books", "output/global.schema.json", Op::Update, Pending, None, 420),
    row("open-books", "…g/tests/test_reading_guide_config.py", Op::Update, Pending, None, 420),
    row("open-books", "…ols/tests/test_collapse_band_span.py", Op::Update, Pending, None, 420),
    row("open-books", "…/stage_b/tests/test_reading_guide.py", Op::Update, Pending, None, 420),
    row("open-books", "…_building/common/preamble/header.tex", Op::Update, Pending, None, 420),
    row("open-books", "…g/tests/test_collapse_band_single.py", Op::Update, Pending, None, 420),
    row("open-books", "…ATTING_RULES/05-url-link-handling.md", Op::Update, Pending, None, 420),
    row("open-books", "…es/reading-guide/golden/expected.tex", Op::Update, Pending, None, 420),
    row("workspace-qdrant-mcp", "…le/wqm-tui/src/widgets/chrome/mod.rs", Op::Update, Pending, None, 480),
    row("workspace-qdrant-mcp", "…odule/wqm-tui/src/panes/cell/sort.rs", Op::Add, Pending, None, 480),
    row("workspace-qdrant-mcp", "…ui/src/views/dashboard/tests/sort.rs", Op::Add, Pending, None, 480),
    row("workspace-qdrant-mcp", "…ui/src/views/dashboard/ingredient.rs", Op::Update, Pending, None, 480),
    row("workspace-qdrant-mcp", "…/wqm-tui/src/widgets/chrome/keyed.rs", Op::Add, Pending, None, 480),
    row(".config", "figlet/uskata.flc", Op::Add, Pending, Some(747), 1260),
    row(".config", "…ts/default/nextcloud_log.json.sample", Op::Add, Pending, Some(3379), 1260),
    row(".config", "…89401/tools/get-svg-component.js.map", Op::Add, Pending, Some(472371), 1260),
    row(".config", "…ant/qdrant-web-ui/public/logo192.png", Op::Add, Pending, Some(6144), 1260),
    row(".config", "…30581ac/toggle-minimal-effect.js.map", Op::Add, Pending, Some(2560), 1260),
    row(".config", "…-30f96a3c9f83/cmd-show-status.js.map", Op::Add, Pending, Some(357990), 1260),
    row(".config", "figlet/5lineoblique.flf", Op::Add, Pending, Some(8806), 1260),
    row(".config", "…42f1-baa9-1541966a1e00/custom.js.map", Op::Add, Pending, Some(4508877), 1260),
    row(".config", "xm1/texmakerapp.ini", Op::Add, Pending, Some(25), 1260),
    row(".config", "…ormats/default/uwsgi_log.json.sample", Op::Add, Pending, Some(3686), 1260),
    row(".config", "…5bd105e/tools/run-applescript.js.map", Op::Add, Pending, Some(1434), 1260),
    row(".config", "…memory/MEMORY.md.bak-20260825-130321", Op::Add, Pending, Some(2048), 1260),
    row(".config", "…d2570668dd6/search-userstyles.js.map", Op::Add, Pending, Some(897741), 1260),
    row(".config", "…addfadfbaf/assets/osicons/win-11.png", Op::Add, Pending, Some(195), 1260),
    row(".config", "figlet/Bright.flf", Op::Add, Pending, Some(6963), 1260),
    row(".config", "epy/states.db", Op::Add, Pending, Some(28672), 1260),
    row(".config", "figlet/SL Script.flf", Op::Add, Pending, Some(4915), 1260),
    row(".config", "…f4-8a1520c55898/search-photos.js.map", Op::Add, Pending, Some(1153434), 1260),
    row(".config", "…15-b14c-fcb7ed51dbae/timeline.js.map", Op::Add, Pending, Some(3250586), 1260),
    row(".config", "…-8c8d4008c461/markdownpreview.js.map", Op::Add, Pending, Some(1946), 1260),
    row(".config", "…d72b8e/assets/screen-mirror-icon.png", Op::Add, Pending, Some(10752), 1260),
    row(".config", "…-481e-8bb9-44c4c6df53be/index.js.map", Op::Add, Pending, Some(68506), 1260),
    row(".config", "omp/agent/terminal-sessions/ttys003", Op::Add, Pending, Some(135), 1260),
    row(".config", "figlet/henry3d.flf", Op::Add, Pending, Some(9728), 1260),
    row(".config", "weechat/xfer.conf", Op::Add, Pending, Some(981), 1260),
    row(".config", "…95cf92e05cf6/assets/grab-delayed.png", Op::Add, Pending, Some(155341), 1260),
    row(".config", "figlet/Rozzo.flf", Op::Add, Pending, Some(7680), 1260),
    row(".config", "…d5ba06bb/assets/timezones/UTC-12.png", Op::Add, Pending, Some(40858), 1260),
    row(".config", "…/assets/compiled_raycast_swift/swift", Op::Add, Pending, Some(593101), 1260),
    row(".config", "lazygit/catppuccin/flake.lock", Op::Add, Pending, Some(2458), 1260),
    row(".config", "…/common/assets/git/leak-denylist.mcp", Op::Add, Pending, Some(964), 1260),
    row(".config", "herdr/config.toml", Op::Update, Pending, Some(27955), 1260),
    row(".config", "…3c87730581ac/toggle-auto-hide.js.map", Op::Add, Pending, Some(3072), 1260),
    row(".config", "…c2f583/assets/codesandbox-bright.png", Op::Add, Pending, Some(671), 1260),
    row(".config", "…nks/tmp-1mmfhhwpdy6hi/mnemopi.db-wal", Op::Add, Pending, Some(2411725), 1260),
    row("open-books", "…ools/tests/test_collapse_codesize.py", Op::Update, Pending, None, 1260),
    row(".config", "…0-44b7-9446-55b52a1cac36/open.js.map", Op::Add, Pending, Some(1020723), 1260),
    row(".config", "figlet/Graffiti.flf", Op::Add, Pending, Some(6554), 1260),
    row(".config", "…/bat/themes/Catppuccin Latte.tmTheme", Op::Add, Pending, Some(64307), 1260),
    row(".config", "…3-c34feb6486ee/assets/arxiv-icon.png", Op::Add, Pending, Some(30925), 1260),
    row("workspace-qdrant-mcp", "…/wqm-tui/src/widgets/chrome/keyed.rs", Op::Update, Pending, Some(4915), 1260),
    row("open-books", "…ace/s335-replay/plan-s335-after.json", Op::Add, Pending, None, 1260),
    row(".config", "…09c22a6dd8/create-docset-link.js.map", Op::Add, Pending, Some(5632), 1260),
    row(".config", "…a4c6/tools/list-workflow-runs.js.map", Op::Add, Pending, Some(3040870), 1260),
    row("open-books", "…lding/common/tools/collapse_align.py", Op::Update, Pending, None, 1260),
    row(".config", "snyk/ls-config-Neovim", Op::Add, Pending, Some(4096), 1260),
    row(".config", "…-4d34ead78777/assets/branch@dark.svg", Op::Add, Pending, Some(339), 1260),
    row(".config", "…kills/tui-test-harness/bin/tui-close", Op::Add, Pending, Some(1434), 1260),
    row("open-books", "…uilding/common/tools/collapse_lex.py", Op::Add, Pending, None, 1260),
    row(".config", "…64c-4dba-ba77-b1cb543256d6/ai.js.map", Op::Add, Pending, Some(278733), 1260),
    row(".config", "…de-616a22387eae/assets/icon-stop.png", Op::Add, Pending, Some(231), 1260),
    row(".config", "…d5ba06bb/assets/timezones/UTC+10.png", Op::Add, Pending, Some(42394), 1260),
    row(".config", "…nsupported-simulators.command.js.map", Op::Add, Pending, Some(9830), 1260),
    row(".config", "…46b-a8f9-d710b41aa36d/grammar.js.map", Op::Add, Pending, Some(3355443), 1260),
    row(".config", "…a-a5c1-1c26f5b329a9/assets/issue.png", Op::Add, Pending, Some(1126), 1260),
    row(".config", "…-4750-8b4a-81e748f2c33a/index.js.map", Op::Add, Pending, Some(316621), 1260),
    row(".config", "…12a465/fan-add-event-editable.js.map", Op::Add, Pending, Some(1153434), 1260),
    row(".config", "figlet/Isometric1.flf", Op::Add, Pending, Some(12800), 1260),
    row(".config", "Code/TransportSecurity", Op::Add, Pending, Some(5427), 1260),
    row(".config", "…ts/prompts-library-db.0.mdb/data.mdb", Op::Add, Pending, Some(12288), 1260),
    row(".config", "…b-4c8d-9bfd-8c8d4008c461/auto.js.map", Op::Add, Pending, Some(1843), 1260),
    row(".config", "…-95bf-4d34ead78777/assets/branch.svg", Op::Add, Pending, Some(339), 1260),
    row(".config", "…37fc20bd0c2/formatToJsonValue.js.map", Op::Add, Pending, Some(1153434), 1260),
    row("open-books", "…workspace/s335-replay/plan-s335.json", Op::Add, Pending, None, 1260),
    row("open-books", "…ools/tests/test_collapse_identity.py", Op::Update, Pending, None, 1260),
    row("workspace-qdrant-mcp", "…odule/wqm-tui/src/panes/cell/sort.rs", Op::Update, Pending, Some(2458), 1260),
    row("open-books", "…ng/common/tools/collapse_identity.py", Op::Update, Pending, None, 1260),
    row("open-books", "…tools/tests/test_collapse_aligned.py", Op::Update, Pending, None, 1260),
    row(".config", "…-be8a-f740b3f1656a/assets/remote.svg", Op::Add, Pending, Some(413), 1260),
    row(".config", "…8857d5ba06bb/assets/command-icon.png", Op::Add, Pending, Some(94003), 1260),
    row(".config", "…/toggle-stickies-float-on-top.js.map", Op::Add, Pending, Some(1153434), 1260),
    row(".config", "…46-55b52a1cac36/assets/list-icon.png", Op::Add, Pending, Some(5530), 1260),
    row(".config", "…-87ccac1299a1/tools/open-path.js.map", Op::Add, Pending, Some(795), 1260),
    row("open-books", "…n/tools/tests/test_collapse_delta.py", Op::Update, Pending, None, 1260),
    row(".config", "eza/eza-themes/test_dir/song.flac", Op::Add, Pending, Some(0), 1260),
    row(".config", "…-4724-afb9-9835e315ba92/index.js.map", Op::Add, Pending, Some(538624), 1260),
    row(".config", "…-616a22387eae/assets/docker-icon.png", Op::Add, Pending, Some(66150), 1260),
    row(".config", "figlet/JS Cursive.flf", Op::Add, Pending, Some(3584), 1260),
    row(".config", "figlet/miniwi.flf", Op::Add, Pending, Some(2765), 1260),
    row(".config", "…60725f2a39/assets/status-loading.png", Op::Add, Pending, Some(351), 1260),
    row(".config", "…ormats/default/caddy_log.json.sample", Op::Add, Pending, Some(3686), 1260),
    row(".config", "…583db3f4f/word-synonym-search.js.map", Op::Add, Pending, Some(275763), 1260),
    row(".config", "codex/common/assets/git/leak-denylist", Op::Add, Pending, Some(2867), 1260),
    row(".config", "…r/sessions/Main/session-history.json", Op::Update, Pending, Some(590029), 1260),
    row(".config", "…za-themes/imgs/catppuccin-frappe.png", Op::Add, Pending, Some(191386), 1260),
    row(".config", "…onfigs/default/uk-keymap.json.sample", Op::Add, Pending, Some(386), 1260),
    row("open-books", "…n/tools/tests/test_collapse_alias.py", Op::Update, Pending, None, 1260),
    row(".config", "…s/default/lnav_debug_log.json.sample", Op::Add, Pending, Some(3686), 1260),
    row(".config", "…efault/github_events_log.json.sample", Op::Add, Pending, Some(5939), 1260),
    row("open-books", "…e/s335-replay/plan-s335-arm-tok.json", Op::Add, Pending, None, 1260),
    row(".config", "…s/btop/themes/catppuccin_mocha.theme", Op::Add, Pending, Some(2253), 1260),
    row(".config", "…cb543256d6/tools/list-folders.js.map", Op::Add, Pending, Some(253747), 1260),
    row(".config", "…ed4e9d8ed1/assets/microsoftAzure.png", Op::Add, Pending, Some(72090), 1260),
    row("open-books", "…lding/common/tools/collapse_alias.py", Op::Update, Pending, None, 1260),
    row(".config", "…default/lnav-breakpoint-handler.lnav", Op::Add, Pending, Some(428), 1260),
    row(".config", "figlet/Broadway.flf", Op::Add, Pending, Some(15053), 1260),
    row(".config", "figlet/Shadow.flf", Op::Add, Pending, Some(14541), 1260),
    row(".config", "…5d-bd84-eb02c43eb904/releases.js.map", Op::Add, Pending, Some(741069), 1260),
    row(".config", "…ugins/config/herdr-lazy/plugins.list", Op::Add, Pending, Some(577), 1260),
    row(".config", "…ts/default/procstate_log.json.sample", Op::Add, Pending, Some(974), 1260),
    row(".config", "…62-95cf92e05cf6/assets/load-file.png", Op::Add, Pending, Some(118784), 1260),
    row(".config", "…a6-fca9-7000-b758-d7179632b993.jsonl", Op::Add, Pending, Some(4301), 1320),
    row(".config", "…sets/compiled_raycast_swift/contacts", Op::Add, Pending, Some(690790), 1320),
    row(".config", "…12a465/fan-quick-add-reminder.js.map", Op::Add, Pending, Some(1153434), 1320),
    row(".config", "…a7-da8b-7000-89e7-4655e02d9e03.jsonl", Op::Add, Pending, Some(4198), 1320),
    row(".config", "figlet/646-cn.flc", Op::Add, Pending, Some(5018), 1320),
    row(".config", "…-9bfd-8c8d4008c461/colortools.js.map", Op::Add, Pending, Some(1843), 1320),
    row(".config", "…daddfadfbaf/assets/osicons/linux.png", Op::Add, Pending, Some(814), 1320),
    row(".config", "figlet/Script.flf", Op::Add, Pending, Some(16896), 1320),
    row(".config", "figlet/Double Shorts.flf", Op::Add, Pending, Some(3482), 1320),
    row(".config", "…465/fan-add-reminder-editable.js.map", Op::Add, Pending, Some(1153434), 1320),
    row(".config", "eza/eza-themes/test_dir/file.mp4", Op::Add, Pending, Some(0), 1320),
    row(".config", "lnav/cmd.history", Op::Add, Pending, Some(0), 1320),
    row(".config", "…6b-a8f9-d710b41aa36d/friendly.js.map", Op::Add, Pending, Some(3355443), 1320),
    row(".config", "…open-books/memory/MEMORY.md.bak-s106", Op::Add, Pending, Some(15360), 1320),
    row(".config", "…formats/default/page_log.json.sample", Op::Add, Pending, Some(2458), 1320),
    row(".config", "figlet/Mini.flf", Op::Add, Pending, Some(9114), 1320),
    row(".config", "figlet/646-es2.flc", Op::Add, Pending, Some(5018), 1320),
    row(".config", "…-9f40-75fcb039a4c6/assets/vscode.svg", Op::Add, Pending, Some(483), 1320),
    row(".config", "…e8-4373-8162-95cf92e05cf6/ssw.js.map", Op::Add, Pending, Some(2867), 1320),
    row(".config", "figlet/amcthin.flf", Op::Add, Pending, Some(7066), 1320),
    row(".config", "…tools/xcode-add-swift-package.js.map", Op::Add, Pending, Some(461517), 1320),
    row(".config", "…96a3c9f83/cmd-clean-snapshots.js.map", Op::Add, Pending, Some(32870), 1320),
    row(".config", "…8162-95cf92e05cf6/assets/uploads.png", Op::Add, Pending, Some(140902), 1320),
    row(".config", "…-459a-a5c1-1c26f5b329a9/index.js.map", Op::Add, Pending, Some(2726298), 1320),
    row(".config", "…f-885c-acf4ebcf19b1/summarize.js.map", Op::Add, Pending, Some(5033165), 1320),
    row(".config", "figlet/koi8r.flc", Op::Add, Pending, Some(2048), 1320),
    row(".config", "…c6f6c93/paste-latest-otp-code.js.map", Op::Add, Pending, Some(274637), 1320),
    row(".config", "zsh/themes/bat/assets/macchiato.webp", Op::Add, Pending, Some(64205), 1320),
    row(".config", "…96de586cef/see-important-mail.js.map", Op::Add, Pending, Some(9017754), 1320),
    row(".config", "…507-665fd0e11784/conversation.js.map", Op::Add, Pending, Some(1782579), 1320),
    row(".config", "…6-8b11-261e5c6f6c93/open-chat.js.map", Op::Add, Pending, Some(497869), 1320),
    row(".config", "…/default/zap_console_log.json.sample", Op::Add, Pending, Some(1434), 1320),
    row(".config", "…d-9bfd-8c8d4008c461/yaml2json.js.map", Op::Add, Pending, Some(1843), 1320),
    row(".config", "…84-eb02c43eb904/assets/icon@dark.png", Op::Add, Pending, Some(157286), 1320),
    row(".config", "…9f40-75fcb039a4c6/my-projects.js.map", Op::Add, Pending, Some(3250586), 1320),
    row(".config", "…225debe4eeb85218468e5f6fd6466f572013", Op::Add, Pending, Some(67), 1320),
    row(".config", "figlet/Doom.flf", Op::Add, Pending, Some(7578), 1320),
    row(".config", "…d48a98/assets/1password-settings.png", Op::Add, Pending, Some(244122), 1320),
    row(".config", "…-unsupported-runtimes.command.js.map", Op::Add, Pending, Some(10035), 1320),
    row(".config", "figlet/Efti Wall.flf", Op::Add, Pending, Some(7270), 1320),
    row(".config", "…/create-swift-package.command.js.map", Op::Add, Pending, Some(1572864), 1320),
    row(".config", "…git/catppuccin/assets/macchiato.webp", Op::Add, Pending, Some(48435), 1320),
    row(".config", "figlet/Computer.flf", Op::Add, Pending, Some(6656), 1320),
    row(".config", "…-211216310ee5/recentDownloads.js.map", Op::Add, Pending, Some(836403), 1320),
    row(".config", "…s/default/default-keymap.json.sample", Op::Add, Pending, Some(9626), 1320),
    row(".config", "…ills/tui-test-harness/bin/tui-launch", Op::Add, Pending, Some(5939), 1320),
    row(".config", "figlet/Lean.flf", Op::Add, Pending, Some(28570), 1320),
    row(".config", "…/search-code-snippets.command.js.map", Op::Add, Pending, Some(1572864), 1320),
    row(".config", "…-4851-a6d3-c34feb6486ee/index.js.map", Op::Add, Pending, Some(2831155), 1320),
    row(".config", "codex/common/scripts/mcp/context7", Op::Add, Pending, Some(136), 1320),
    row("workspace-qdrant-mcp", "…ves/prd-workspace/audit_report_r1.md", Op::Add, InProgress, Some(23757), 7200),
    row("workspace-qdrant-mcp", "…ves/prd-workspace/audit_report_r5.md", Op::Add, InProgress, Some(12902), 7200),
    row("workspace-qdrant-mcp", "…-workspace/audit-r5-consolidation.md", Op::Add, InProgress, Some(15155), 7200),
    row("workspace-qdrant-mcp", "…s/prd-workspace/audit-r3-security.md", Op::Add, InProgress, Some(8192), 7200),
    row("workspace-qdrant-mcp", "…workspace/audit-r3-implementation.md", Op::Add, InProgress, Some(14234), 7200),
    row("workspace-qdrant-mcp", "…rd-workspace/designer-response-r7.md", Op::Add, InProgress, Some(10854), 7200),
    row("workspace-qdrant-mcp", "…ves/prd-workspace/audit-r4-domain.md", Op::Add, InProgress, Some(10854), 7200),
    row("workspace-qdrant-mcp", "…rchives/prd-workspace/audit-r4-ux.md", Op::Add, InProgress, Some(7680), 7200),
    row("workspace-qdrant-mcp", "…ves/prd-workspace/audit_report_r2.md", Op::Add, InProgress, Some(21299), 7200),
    row("workspace-qdrant-mcp", "…s/prd-workspace/audit-r4-security.md", Op::Add, InProgress, Some(8090), 7200),
    row("PlotSwift", "…ces/PlotSwift/Axes+Decorations.swift", Op::Add, Failed, Some(8499), 68400),
    row("PlotSwift", "…lotSwiftTests/AnnotationsTests.swift", Op::Add, Failed, Some(4403), 68400),
    row("PlotSwift", "…otSwiftTests/SeabornPlotsTests.swift", Op::Add, Failed, Some(10650), 68400),
];
