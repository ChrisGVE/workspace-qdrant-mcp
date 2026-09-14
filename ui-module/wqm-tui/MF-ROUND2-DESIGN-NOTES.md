# Modal framework — round 2 design notes

Working log for round 2 of the wqm-tui modal framework, answering Chris's rulings of
2026-09-14 19:05 (plus the 19:14 addition). One section per item of his message, in his order.

Every number here comes from a test that prints it. Re-run rather than quote:

```
cargo test --all-features -p wqm-tui tokens:: -- --nocapture --test-threads=1
```

Frames: `cargo pantry list` → group **`Modal Framework R2`** (35 variants), and
`cargo run -p wqm-tui --features png-capture,tui-pantry --example round2_frames -- out/`
for the 39 PNGs. The index is `MF-ROUND2-FRAMES.tsv` beside this file.

---

## 0. The headline, before the items

**One change answers items 2 and 4 together, and it is the only thing in round 2 that is
structural rather than cosmetic.** Chris raised the tint to 0.40 *and* called the text
unreadable in one message. Those are cause and effect.

The blend was a straight sRGB mix, which moves **lightness together with hue**. `accent` is a
bright colour on every bundled theme, so pulling the window's dark fill 40% toward it makes the
fill *lighter* — and every text rung above it loses the ground it was standing on. Measured, body
text at rung 85 on the window fill:

| | k=0.00 | k=0.14 | k=0.28 | k=0.40 |
|---|---|---|---|---|
| straight sRGB mix (round 1) | 7.2 | 5.6 | 4.4 | **3.6** |
| lightness held (round 2) | 7.2 | 7.2 | 7.2 | **7.3** |

(Catppuccin Mocha; WCAG 2.2 AA body floor is 4.5:1.) Across the fifteen bundled themes at the
ruled k=0.40: **the straight mix clears the floor on 2 of 15; holding lightness on 13 of 15.**
The two that still fail — both Solarized flavours — fail *untinted* too; their own foreground
sits 3.7:1 and 3.4:1 from their own background, which is the theme's choice and not ours.

`TintBlend::HoldLuminance` mixes `a*`/`b*` at the full strength and keeps `L*` where the rung put
it. The window is as blue at 0.40 as Chris asked for and as dark as rung 15 has always been.

**It also rescues item 3 for free** — see §3. The underline he wanted gone was load-bearing only
*because* the straight mix was compressing the fill that would otherwise have carried the mark.

Frames: **05** (straight, round 1) vs **06** (held, round 2). Same hue, same strength, same
everything else.

---

## 1. Size model — items 0 and 1

### The maximum

`Footprint::Max` = the page inset by **5 columns each side and 5 rows top and bottom**, at any
size, exactly as worded. Measured on the painted buffer rather than on the rect:

| page | window |
|---|---|
| 125 × 34 | 115 × 24 |
| 100 × 30 (the floor) | 90 × 20 |
| 200 × 40 | 190 × 30 |

Round 1's `Footprint` A/B is **dissolved, not decided** — nothing derives a window from the help
window's content any more. Both old arms stay reachable so a frame pair can show what changed.

Frames: **00** (dump at all three sizes).

### The minimum, derived

Not a preference. It is the sum of what the window already promised to draw:

```
border 2  +  top decoration 4  +  bottom decoration 3  +  one content row 1  =  10 rows
border/padding 4  +  the narrowest value column a record will accept 30      =  34 columns
```

So the smallest window is **34 × 10**, and the smallest PAGE that can hold one is **44 × 20**. A
test asserts that one row less leaves a zero-height viewport, which is what makes it a floor.

Frame: **01**, dumped at 44 × 20.

### Below it

`Footprint::window` returns `None` rather than a clamped rect — a caller handed a too-small rect
draws a broken window; a caller handed nothing has to say something instead.

`TooSmall` is what it says: two centred lines on the page, **no box** (item (e): a window's
background is its frame, and this has no background to be a frame of). It names the size needed
*and* the size present, so the reader can act on it rather than drag and guess.

> ⚠ **Found by rendering.** The first version did not fit a small screen: at 30 columns it came
> out as `This window needs a larger scr`, cut mid-word. On an otherwise empty screen a truncated
> sentence is indistinguishable from a crash. It now picks the longest form that fits and drops
> the detail line rather than truncating it.

Frame: **02**, dumped at 40 × 18.

### The two sizing policies

- **Not a drill-down** → `Footprint::Content { cols, rows }`, capped by the maximum and floored
  by the minimum. Frame **03** (a Configuration-like record, 62 × 14).
- **A drill-down** → opens at the maximum, with scrollbars as needed. Frame **04**.

### The horizontal bar

Glyphs are the vertical bar's own pair, rotated:

| | thumb | track |
|---|---|---|
| vertical (existing) | `▐` U+2590 RIGHT HALF BLOCK | `▕` U+2595 RIGHT ONE EIGHTH |
| horizontal (new) | `▄` U+2584 LOWER HALF BLOCK | `▁` U+2581 LOWER ONE EIGHTH |

Same two weights, same half-versus-eighth relationship, same two rungs. A different family
(`━`/`─`) would make two scrollbars read as two mechanisms.

It sits on the bottom row of the viewport and costs that row whether or not the thumb is drawn —
a viewport that grew a row back when the content happened to fit would reflow the moment a filter
changed the row count.

Measured on frame 04 at 125×34: vertical thumb 5 rows on a 14-row track, horizontal thumb 80
columns on a 110-column track.

> ⚠ **Found by the grid dump, invisible in the PNG.** A first cut tripled the table fixture to
> "make it scroll". The fixture is 200 rows; the thirteen visible are the viewport. Tripling took
> the vertical thumb from five rows to **one** — the frame looked more scrollable and said less.

### ⚠ OBJECTION — item 0 contradicts a round-1 decision, and it is Chris's to settle

Round 1 pushed every window **below the page header** (app bar, its rule, the three-row status
block) for two stated reasons: a mathematically centred window lands its top border exactly on
that rule at both 125×34 and the floor — two box-drawing runs of the same weight on one row,
which reads as welded rather than floating — and keeping the status block visible means a modal
never costs the reader the answer to *is anything wrong* (Nielsen #1).

**Item 0 says five rows, full stop. At 125×34 that puts the window's top border on row 5, inside
the status block, which occupies rows 2–5.**

The round-2 footprints follow the ruling literally; the round-1 ones keep their clamp. Compare
frame **00** against round 1's `Footprint: B (framework)`. Two conforming alternatives if he
wants the block back: inset 5 but **never above the header** (the window simply sits lower on
tall terminals), or inset from the *content area* rather than from the screen.

---

## 2. Tint — item 2

Accent (blue on Mocha) at **0.40**, confirmed. The neutral / lavender / 0.14 / 0.28 arms are
retired and not rendered.

The only change is the blend (§0). Frames **06** (Mocha), **07** (Solarized Dark — the thinnest
ladder of the fifteen), **08** (Catppuccin Latte — a light theme, where the ladder runs the other
way and a design measured only on Mocha cannot see it).

---

## 3. Fields — item 3, the readability checkpoint

Chris's 19:14 addition makes this a proposal to react to, not a settled thing.

### The underline goes, and nothing is traded for it

Round 1 defended the underline with a measurement: at k=0.28 the SET fill fell to ΔE 2.6 on
Solarized Dark, one just-noticeable difference, and at 0.40 to **2.1 — below a JND**. So "drop
the underline" and "raise to 0.40" looked incompatible.

They are incompatible **only under the straight mix**, because the SET mark is a *lightness* step
and that mix compresses exactly the axis it is made of. SET fill lift off the window at k=0.40:

| | worst theme | range | below the 2.3 JND |
|---|---|---|---|
| straight mix | Solarized Dark **2.1** | 2.1 – 5.4 | 1 of 15 |
| lightness held | Solarized Dark **4.1** | 4.1 – 9.1 | **0 of 15** |

So the fill alone carries the SET mark on every bundled theme, and the underline goes on his word
with nothing given up.

**What does not change is `NO_COLOR`.** There the ladder collapses onto four slots and rung 15 and
rung 22 land on the same one, whatever the blend does — *which fields may I change* would be
carried by colour alone, which r06 #8 forbids. `SetMark::FillWithFallback` puts the underline back
**only there**: invisible in every frame Chris judges, present in every frame he cannot.

Frames **10** (round 1, ruled and filled) vs **11** (round 2, fill alone); **33** is the same
frame under `NO_COLOR` with the underline back.

### ⚠ OBJECTION — "the font of the selected line becomes black" is not a foreground change

The active field's fill is rung 35, and **rung 35 is a middle grey — bad for black and for white
at the same time.** On the surface round 2 actually paints:

| theme | black on rung 35 | the lighter end on rung 35 |
|---|---|---|
| Catppuccin Mocha | **3.9:1** | 4.1:1 |
| Solarized Dark | 2.7:1 | 3.2:1 |
| One Dark Pro | 3.5:1 | 3.7:1 |

Black clears the body floor on 6 of 15 themes there, the light end on 4. So his instruction cannot
be carried out by changing a foreground — **the fill has to move.**

`FieldRungs::BlackText` moves it: the lowest rung at or above 35 on which black clears 4.5:1.
Per theme, because the ladder is a curve — **30** on Nord, **40** on Mocha, **63** on Solarized
Dark. A single fixed rung would have to be 63 to serve all fifteen, and on Mocha that is brighter
than the body text beside it.

The two-tier scheme survives on every theme: the POINT still separates from the SET by more than
the SET lifts off the window, which is what keeps *"these are the doors"* and *"you are standing
in this one"* two marks rather than two shades.

**VIEW mode already had black-bold** on the cursor row (ruling 7), so nothing changes there —
frame **09**. If his 19:05 wording meant the view-mode row rather than the edit-mode active
field, that half is already done and only the edit-mode half above is new.

### Radio

- **One row** (frame **12**): the active button **bold**, the rest normal; `h`/`l` or ←/→ move
  within the field.
- **A column** (frame **13**): one choice per row, **the whole column takes the highlight**,
  `j`/`k` or ↑/↓ move within it.

Separate `Value` variants rather than a flag, because the two differ in three things at once —
layout, highlight extent, and the keys that move inside them — and a flag would have left the
keymap to be inferred somewhere else.

### Drop-down

Frames **14** (open) and **15** (after typing `ts`).

No frame. Current value **first**, everything else **sorted** under it, cursor on the top row.
Typing fuzzy-filters by **subsequence** (`tsf` finds `tree-sitter/function`) — a substring match
would refuse that and the reader would conclude the filter is broken rather than strict.

The three changes are one flag, not three: sorting without moving the cursor would leave the
cursor on whatever sorted into the current value's old index, which is a different value.

### Tick box

Unchanged — `[✓] yes` / `[ ] no`, shape not colour (r06 #8), Space toggles. Visible in every
record frame.

### Cursor shapes

| mode | shape | blink |
|---|---|---|
| vim NORMAL | reversed block on a character | no — frame **17** |
| vim VISUAL | reversed block over the range | yes |
| vim INSERT | `▏` bar between two characters | yes — frame **16** |
| conventional / emacs | the terminal's own cursor | terminal's — frame **18** |

**How blink is proxied in a still: it is not proxied.** The real `SLOW_BLINK` attribute is set,
the one a terminal honours. A still cannot show it and a third invented glyph would be a shape the
running program never draws — so the frame carries it **in the cell**, to be read from a grid dump
and not from a PNG. Same division as wqm#283, where the `●` glyph is right in the dump and missing
from the capture.

Under the conventional keymap **nothing is painted** — the terminal owns the cursor and a painted
one would be a second cursor beside the real one. A headless frame therefore shows no caret, which
is the honest depiction rather than a gap.

### Selected-text colour — A/B, and A is the recommendation

Today selection is `Modifier::REVERSED`, which is **not a colour**: it inverts whatever is under
it, so inside an edit field it comes out wearing the active field's own fill. A colour has to be
named whatever else is decided.

Measured on the surfaces the widget paints, under the proposed look:

| | ΔE off the active fill | text on it | fails | collides with the cursor hue |
|---|---|---|---|---|
| **A** rung-19 inversion, no hue | 8.7 – 26.5 | 4.3 – 10.5:1 | 2/15 | — |
| **B** theme `secondary` | 14.2 – 127.4 | 3.4 – 10.7:1 | 2/15 | **13/15** |

Both are legible. **B's cost is not a ratio**: on every non-Catppuccin bundled theme `secondary`
resolves to the same value as the data cursor's hue, so a text selection and the row cursor would
wear one colour between them. There is no tenth hue spare — `info` is the selector's, the health
three are spoken for, and `accent` is now the window's own tint.

**Recommendation: A.** It spends no hue and reuses the selected-row tint at a second scale, which
is the vocabulary being consistent rather than the palette being short.

Frames **20** (A) and **21** (B).

---

## 4. Readability — item 4

### The third column loses its band

Frames **22** (band, round 1) vs **23** (no band, round 2). Round 2's default. It is
informational, it does not need a region of its own, and the text now stands on the window like
everything else.

### Headers are optional, consistently

Frame **24** shows a record with no third column and therefore no header row at all — that row
goes back to data. Columns one and two never carried titles and still do not.

### The ΔE / contrast table — the defect, and what is left of it

**ΔE is the wrong instrument here and this is why there are two.** ΔE has no direction — the
crate's own `rung_of` already carries that warning — and it is dominated by hue and chroma, which
contribute nothing to reading an 8×13 glyph. A *surface against a surface* stays ΔE's; *text
against the surface under it* is a ratio of light, so it is WCAG's.

Text on the window fill, accent at 0.40, **lightness held** (ΔE then WCAG ratio; `x` = below the
4.5:1 body floor):

| theme | faint (50) — 3rd col | muted (62) — help + read-only | row (75) | normal (85) |
|---|---|---|---|---|
| Dracula | 38.4 · 3.3 x | 50.5 · 4.5 | 63.7 · 6.2 | 73.4 · 7.7 |
| One Dark Pro | 26.7 · 2.3 x | 33.3 · 3.0 x | 40.4 · 3.8 x | 46.0 · 4.6 |
| Nord | 31.7 · 2.9 x | 41.3 · 3.9 x | 51.8 · 5.3 | 59.6 · 6.5 |
| Catppuccin Mocha | 33.0 · 3.1 x | 42.2 · 4.3 x | 52.0 · 5.8 | 59.7 · 7.3 |
| Catppuccin Latte | 27.9 · 2.0 x | 35.0 · 2.8 x | 43.8 · 4.0 x | 51.2 · 5.3 |
| Gruvbox Dark | 38.5 · 2.9 x | 44.6 · 4.0 x | 52.4 · 5.4 | 59.0 · 6.7 |
| Gruvbox Light | 38.4 · 2.4 x | 47.9 · 3.5 x | 59.8 · 5.4 | 68.9 · 7.5 |
| Tokyo Night | 30.8 · 3.0 x | 40.0 · 4.1 x | 50.0 · 5.6 | 57.6 · 7.0 |
| Solarized Dark | 24.8 · 1.9 x | 30.5 · 2.4 x | 36.6 · 3.0 x | 41.5 · 3.6 x |
| Solarized Light | 23.7 · 1.7 x | 28.2 · 2.2 x | 34.2 · 2.7 x | 39.5 · 3.4 x |
| Monokai Pro | 41.6 · 3.3 x | 51.3 · 4.6 | 61.7 · 6.3 | 69.9 · 7.8 |
| Rosé Pine | 35.0 · 3.4 x | 45.4 · 4.8 | 56.2 · 6.7 | 64.6 · 8.4 |
| Kanagawa | 33.3 · 3.0 x | 44.8 · 4.2 x | 56.2 · 5.7 | 65.2 · 7.2 |
| Everforest | 28.2 · 2.4 x | 35.1 · 3.1 x | 43.5 · 4.1 x | 50.3 · 4.9 |
| Cyberpunk | 39.5 · 3.8 x | 52.0 · 5.7 | 65.3 · 8.2 | 75.5 · 10.6 |

**Read the ΔE column and the ratio column against each other.** `faint` sits ΔE 24–42 off the
fill — enormous by any surface standard — and is unreadable on all fifteen themes. That is the
whole argument for the second instrument.

Holding lightness fixes the **body baseline** (13/15). It does nothing for the rungs beneath it,
and those are exactly the three surfaces Chris named: the third column is `faint` (50), the help
labels and a read-only value are both `muted` (62). Those were under the floor *before any tint
existed*, so no blend was ever going to be their cure.

### The proposal: fewer rungs inside a window, not brighter ones

An emphasis ladder makes some text quieter on purpose; a contrast floor says all text must be
readable. Sliding one rung up lands it on the next. The ladder has eleven rungs because a
**screen** has room for that many degrees of emphasis — and a window is ~24 rows where nothing is
scenery.

So **inside a window** `faint` and `muted` collapse onto a single quiet rung, **derived per theme**
against the fill the window is actually painted (floor 75, ceiling 80 so it stays visibly under
the baseline):

- legible on **11 of 15** themes, against **0 of 15** on the ladder;
- each of the four that still fail comes within **10%** of that theme's own baseline — two are the
  Solarized flavours whose foreground is under the floor before we draw anything, and the other two
  (One Dark Pro 4.18, Everforest 4.49) miss 4.5 by less than a tenth of a ratio point.

It is a **scope**, not a parameter, for the reason `tokens::modal` is one: the rule has to reach
the container's help rows, the record's third column and whatever a later view puts inside a
window. A per-widget flag reaches the widgets someone remembered — which is the failure
`tokens::modal`'s own module docs record. The page beneath is drawn outside the scope and keeps
its full ladder.

Help rows still go through `tokens::key_hints` — one producer, unchanged, and the scope is what
makes it ground-aware without a second call site.

---

## 5. Breadcrumb — item (a)

Powerline segments: previous crumbs share **one run** of the modal hue at full saturation with the
separator drawn *inside* it; the current crumb sits on a lighter run and is bold; each run closes
onto the next with the separator in the **outgoing** colour on the **incoming** one — which is
what makes a powerline segment read as one arrow rather than two blocks with a mark between them.

Frames **25** (depth 1), **26** (depth 2), **27** (depth 3), **28** (round 1's plain trail for
comparison), **34** (the same frame at `ansi16`).

Separator count is asserted, not eyeballed: the glyph is `U+E0B0`, in the private-use area, so it
renders as a **blank** in most text dumps and as a replacement box on a terminal without a patched
font. A human reading the frame cannot count them.

### ⚠ OBJECTION — "written in white" fails on 13 of the 15 bundled themes

White on the full-saturation accent:

| theme | white | black |
|---|---|---|
| **Catppuccin Mocha** (what the harness paints) | **1.6:1** | 10.6:1 |
| One Dark Pro | 1.4:1 | — |
| Cyberpunk | 1.3:1 | — |
| Solarized Light | 1.3:1 | — |

Mocha's `accent` is a *light* blue. Across the fifteen the accent lands on both sides of the
middle, so **no single text colour is legible on all of them.**

**Conforming alternative, and it is what is rendered:** the instruction is kept as a *role* — the
crumb text is whichever end of the ladder can be read where it lands. That **is** white wherever
the accent is dark, and is black where white would have been unreadable.

### ⚠ Second objection — on some themes neither end works

Dracula's accent sits mid-range: its best case is **4.2:1**, under the floor for black *and* for
white. `contrast::legible_ground` keeps the hue and the chroma exactly as the theme named them and
moves only `L*` until the text clears — the same trade the tint makes, in the other direction.
Chris's wording names **saturation**, which is the axis left untouched.

### Degradation

Nerd fonts are a **font**, not an encoding. Nothing in the program can detect a terminal without a
patched font, so `CrumbStyle::Plain` stays as the fallback a configuration would select — not
something the renderer guesses at. At `ansi16` (frame 34) the glyphs survive and the hue
separation collapses onto slots.

---

## 6. Edit banner — item (b)

Gone. The columnar change is the indication. Frame **29** is in edit mode and nothing on the title
row says so; the test asserts both halves, because a frame that had merely forgotten to enter edit
mode would satisfy the first.

---

## 7. Confirm over the window — item (c)

Frame **30**. The window beneath the guard gets **the same treatment the page gets** — the existing
`tokens::ModalScope`, entered one level deeper. That is the whole implementation: one rule applied
at one more level, rather than a second rule meaning the same thing. Its hues go, its bright text
rungs collapse to muted, and the guard on `Layer2` stands out because it is outside the scope.

No A/B: there is only one defensible strength, because "greyish but readable" is already a solved
problem in this crate and inventing a second answer to it would be the drift the one-blend rule
exists to prevent.

---

## 8. Border — item (e)

Frames **31** (bordered) vs **32** (spacing only). The same rect, the same fill, the same padding,
no glyphs — **the content does not move by a column**, which is asserted cell-for-cell so the pair
differs in ink alone and Chris is judging one layout rather than two.

---

## Open questions, each as a frame pair

| # | question | frames |
|---|---|---|
| Q1 | Does the 5-row inset earn covering a row of the status block? (§1's objection) | **00** vs round 1's `Footprint: B (framework)` |
| Q2 | Selected-text colour: no hue, or a hue that collides with the cursor on 13/15? | **20** vs **21** |
| Q3 | Did "the font of the selected line becomes black" mean the VIEW cursor row (already done) or the EDIT active field (new, and it moves the fill)? | **09** vs **11** |
| Q4 | Third column: band or no band — is the no-band version quiet enough to still read as reference? | **22** vs **23** |

---

## Methodology notes, for whoever picks this up

- **Judge colour from the PNG, shape and attributes from the grid dump.** `png-capture` drops the
  `●` radio glyph (wqm#283), the powerline separator is invisible in a text view, and no still
  image can show `SLOW_BLINK`. Both instruments read the *same* pantry variant, so they cannot
  disagree about what was drawn — only about what each can show.
- **`str::find` answers in bytes.** A frame test read the window's left margin over a border row
  made of three-byte box glyphs and reported 15 for a margin of 5 — and would have gone on
  reporting 15 however wrong the frame became. That is the render trap this design's own rules
  name, hit inside the instrument meant to check for it.
- **Scope a buffer search to the window.** Counting underlines over the whole screen found 27,
  every one a table *header*, which this crate underlines by ruling. Searching the whole screen
  for a scrollbar glyph passes on the page's own bar while the window has none.
- **Round 2 adds five process-global design brackets** (`TintBlend`, `FieldRungs`, `SetMark`,
  `SelectedText`, `WindowText`). A test that sets one without restoring it repaints every test
  after it — this bit three times in one session. One `Restore` guard names all of them, and a new
  bracket belongs in that tuple before it belongs anywhere else.
- **Both lines put a theme in force.** `Palette::set(Palette::Bundled)` *and* `set_theme`.
  Round 1's equivalent example set only the second, which renders the slot fallbacks: a magenta
  border and every tint frame identical.
