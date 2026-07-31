//! Asking the terminal what colours it is actually painting with.
//!
//! [`crate::tokens::Palette::Derived`] generates its neutrals by interpolating between the
//! terminal's own background and foreground, so it needs those two colours as concrete
//! values. There is no portable API for that — the only way to ask is the XTerm operating
//! system command: write `OSC 11 ; ? ST` and the terminal answers with its background,
//! `OSC 10 ; ?` with its foreground.
//!
//! # Why this talks to `/dev/tty` and not to stdin
//!
//! The reply arrives on the terminal's input stream. Reading it off `stdin` would mean
//! competing with whatever else is reading stdin — in a `cargo pantry` session that is the
//! event loop, and a stolen byte is a swallowed keypress. Opening `/dev/tty` gives this
//! module its own descriptor onto the same terminal, so the query is invisible to the rest
//! of the program.
//!
//! # Why it can always fail, and why that is fine
//!
//! Not every terminal answers (`OSC 11` is widely but not universally supported), and there
//! is no terminal at all when output is a pipe — which is exactly the case for
//! `cargo pantry dump`. Every entry point here returns [`Option`], and the caller is
//! expected to fall back to a palette that needs no detection. For the headless case the
//! environment overrides below make the derived palette reproducible without a terminal.

use std::os::fd::{AsRawFd, RawFd};
use std::time::{Duration, Instant};
use std::{env, fs::File, io::Read, io::Write};

use ratatui::style::Color;

/// An 8-bit-per-channel colour, as the terminal reports it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Parses `#rrggbb` (with or without the `#`). Used by the environment overrides.
    pub fn parse_hex(s: &str) -> Option<Self> {
        let s = s.trim().trim_start_matches('#');
        if s.len() != 6 {
            return None;
        }
        let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
        Some(Self::new(byte(0)?, byte(2)?, byte(4)?))
    }

    /// The channels behind a [`Color`], or [`None`] where there are none to read.
    ///
    /// A slot or an index is a *reference* to a colour the terminal owns, not a colour — so
    /// there is nothing here to interpolate between, and the caller has to say what it wants
    /// done about that rather than being handed a plausible guess. Every bundled theme is RGB
    /// (`theme_sheet::every_theme_gives_ten_distinct_roles_and_four_ordered_neutrals` asserts
    /// it), so in practice this is `Some` for every theme that ships.
    pub const fn from_color(color: Color) -> Option<Self> {
        match color {
            Color::Rgb(r, g, b) => Some(Self::new(r, g, b)),
            _ => None,
        }
    }
}

/// The two ends of the terminal's own contrast range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Endpoints {
    pub background: Rgb,
    pub foreground: Rgb,
}

/// The environment overrides, checked before any terminal is touched.
///
/// These exist so a derived palette can be rendered where no terminal will answer — CI, a
/// piped `cargo pantry dump`, an agent capturing frames. Setting them is also the way to
/// preview *someone else's* theme without switching to it.
const ENV_BG: &str = "WQM_TUI_TERM_BG";
const ENV_FG: &str = "WQM_TUI_TERM_FG";

/// How long to wait for the terminal to answer. A terminal that intends to reply does so
/// within a round trip; anything longer means it does not implement the query, and waiting
/// only delays startup.
const REPLY_TIMEOUT: Duration = Duration::from_millis(150);

/// Both endpoints, from the environment if set, otherwise from the terminal.
///
/// The environment wins deliberately: an override is an explicit instruction, and it is the
/// only mechanism available when there is no terminal to ask.
pub fn detect() -> Option<Endpoints> {
    if let Some(endpoints) = from_env() {
        return Some(endpoints);
    }
    query_terminal()
}

fn from_env() -> Option<Endpoints> {
    let bg = Rgb::parse_hex(&env::var(ENV_BG).ok()?)?;
    let fg = Rgb::parse_hex(&env::var(ENV_FG).ok()?)?;
    Some(Endpoints {
        background: bg,
        foreground: fg,
    })
}

/// Asks the terminal directly. Returns `None` unless *both* colours come back, because a
/// ladder interpolated between a real endpoint and a guessed one would misreport which
/// theme it is honouring.
fn query_terminal() -> Option<Endpoints> {
    let mut tty = File::options()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .ok()?;
    let fd = tty.as_raw_fd();

    // SAFETY: `fd` is owned by `tty` and stays open for the whole block. `Termios` is
    // written by `tcgetattr` before it is read.
    let saved = unsafe { raw_mode(fd)? };
    let result = exchange(&mut tty);
    // SAFETY: restoring the settings captured above, on the same descriptor.
    unsafe { restore_mode(fd, &saved) };

    result
}

/// Writes both queries, then reads until the replies arrive or the deadline passes.
fn exchange(tty: &mut File) -> Option<Endpoints> {
    // `ST` is written as BEL: every terminal that understands OSC accepts it, and it is a
    // single byte, so a truncated read cannot split the terminator.
    tty.write_all(b"\x1b]11;?\x07\x1b]10;?\x07").ok()?;
    tty.flush().ok()?;

    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    let deadline = Instant::now() + REPLY_TIMEOUT;

    while Instant::now() < deadline {
        match tty.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                // Both replies are in hand as soon as each prefix has been seen; reading
                // further would block until the timeout for no gain.
                if let (Some(bg), Some(fg)) = (parse_reply(&buf, 11), parse_reply(&buf, 10)) {
                    return Some(Endpoints {
                        background: bg,
                        foreground: fg,
                    });
                }
            }
            // Raw mode is configured to return rather than block, so "nothing yet" arrives
            // as an error; the deadline is what ends the loop.
            Err(_) => continue,
        }
    }
    None
}

/// Pulls one `OSC <code> ; rgb:RRRR/GGGG/BBBB` reply out of the accumulated bytes.
///
/// The components are 16-bit per the spec (terminals answer `1e1e`, not `1e`), so each is
/// scaled down by taking its high byte. Some terminals reply with 8- or 12-bit components
/// instead, which is why the width is measured rather than assumed.
fn parse_reply(buf: &[u8], code: u8) -> Option<Rgb> {
    let text = String::from_utf8_lossy(buf);
    let marker = format!("]{code};rgb:");
    let rest = &text[text.find(&marker)? + marker.len()..];
    let end = rest
        .find(|c: char| c != '/' && !c.is_ascii_hexdigit())
        .unwrap_or(rest.len());
    let mut parts = rest[..end].split('/');

    let mut component = || -> Option<u8> {
        let hex = parts.next()?;
        let value = u32::from_str_radix(hex, 16).ok()?;
        // Normalise whatever width was sent onto 8 bits.
        Some(match hex.len() {
            1 => (value * 0x11) as u8,
            2 => value as u8,
            3 => (value >> 4) as u8,
            _ => (value >> 8) as u8,
        })
    };
    Some(Rgb::new(component()?, component()?, component()?))
}

// --- termios ------------------------------------------------------------------------
// The query only works in raw mode: with echo on, the reply is printed into the user's
// session, and with canonical input on, nothing is readable until a newline that the
// terminal never sends.

/// SAFETY: `fd` must be an open file descriptor onto a terminal.
unsafe fn raw_mode(fd: RawFd) -> Option<libc::termios> {
    let mut saved: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut saved) } != 0 {
        return None;
    }
    let mut raw = saved;
    raw.c_lflag &= !(libc::ICANON | libc::ECHO);
    // Return immediately with whatever has arrived, so the deadline in `exchange` is the
    // only thing that governs how long this takes.
    raw.c_cc[libc::VMIN] = 0;
    raw.c_cc[libc::VTIME] = 1;
    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
        return None;
    }
    Some(saved)
}

/// SAFETY: `fd` must be the descriptor `raw_mode` was called with, still open.
unsafe fn restore_mode(fd: RawFd, saved: &libc::termios) {
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, saved) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_with_and_without_hash() {
        assert_eq!(Rgb::parse_hex("#1e1e2e"), Some(Rgb::new(0x1e, 0x1e, 0x2e)));
        assert_eq!(Rgb::parse_hex("cdd6f4"), Some(Rgb::new(0xcd, 0xd6, 0xf4)));
        assert_eq!(Rgb::parse_hex("#12345"), None);
        assert_eq!(Rgb::parse_hex("zzzzzz"), None);
    }

    #[test]
    fn parses_a_sixteen_bit_osc_reply() {
        let reply = b"\x1b]11;rgb:1e1e/1e1e/2e2e\x07";
        assert_eq!(parse_reply(reply, 11), Some(Rgb::new(0x1e, 0x1e, 0x2e)));
    }

    #[test]
    fn parses_both_replies_out_of_one_buffer() {
        let reply = b"\x1b]11;rgb:1e1e/1e1e/2e2e\x07\x1b]10;rgb:cdcd/d6d6/f4f4\x07";
        assert_eq!(parse_reply(reply, 11), Some(Rgb::new(0x1e, 0x1e, 0x2e)));
        assert_eq!(parse_reply(reply, 10), Some(Rgb::new(0xcd, 0xd6, 0xf4)));
    }

    #[test]
    fn normalises_narrower_components() {
        assert_eq!(
            parse_reply(b"\x1b]11;rgb:1e/1e/2e\x07", 11),
            Some(Rgb::new(0x1e, 0x1e, 0x2e))
        );
    }

    #[test]
    fn missing_reply_is_none_not_a_guess() {
        assert_eq!(parse_reply(b"\x1b]11;rgb:1e1e/1e1e/2e2e\x07", 10), None);
    }
}
