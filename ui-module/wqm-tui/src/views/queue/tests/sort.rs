//! The columns, their sort keys, and what pressing one does.

use super::*;
use crate::panes::cell::{Direction, Sort};

/// The column set is v0.1's, in v0.1's order, with `ID` replaced by `No` — and the sort keys are
/// the ones this screen binds.
///
/// Spelled out here rather than derived from [`frames::columns`], which would be the declaration
/// agreeing with itself. The list IS the ruling, so a change to either has to be a change to
/// both.
#[test]
fn the_columns_are_v01s_in_v01s_order_with_the_ruled_sort_keys() {
    let columns = frames::columns();
    let got: Vec<(&str, Option<char>)> = columns
        .iter()
        .map(|column| (column.title, column.sort_key))
        .collect();
    assert_eq!(
        got,
        vec![
            ("", None),
            ("T", None),
            ("Tenant", Some('e')),
            ("Object", Some('b')),
            ("Type", None),
            ("Op", Some('p')),
            ("Status", Some('u')),
            ("Size", Some('z')),
            ("Age", Some('a')),
        ]
    );
    assert!(
        !got.iter().any(|(title, _)| *title == "ID"),
        "v0.1's hash column is replaced by No, not kept beside it"
    );
}

/// `No` is untitled and offers no key: the reference number is an invariant, not a fact about
/// the row that could be ordered — and the letter it freed (`o`) is the operation selector's.
#[test]
fn the_reference_column_offers_no_key_and_frees_the_o_for_the_selector() {
    let columns = frames::columns();
    assert_eq!(columns[frames::NO].sort_key, None);
    assert_eq!(columns[frames::NO].title, "");
    assert!(
        QUEUE_BOUND_KEYS.contains(&'o'),
        "`o` is bound to the operation selector, which the keyless `No` column freed"
    );
}

/// `T` offers no key: the type selector is gone and the free-text filter covers narrowing by
/// type — and `Type` next door sorts by the same fact when a reader wants it gathered rather
/// than removed.
#[test]
fn the_type_column_offers_no_sort_key_because_the_filter_covers_narrowing_by_type() {
    let columns = frames::columns();
    assert_eq!(columns[frames::T].sort_key, None);
}

/// Chris's *non-ambiguous* requirement, made structural: no column offers a letter twice, and
/// none offers a letter the screen has already bound.
///
/// **Case-insensitive**, and this screen is why. `n` and `N` are two different bindings here, and
/// [`crate::widgets::chrome::keyed_spans`] finds a key in a title without regard to case — so a
/// column offering `N` would light the `n` in its own name.
#[test]
fn every_sort_key_is_unique_and_free_of_the_keys_the_screen_has_bound() {
    let bound: Vec<char> = QUEUE_BOUND_KEYS
        .iter()
        .chain(QUEUE_BOUND_EXTRA.iter())
        .flat_map(|key| key.to_lowercase())
        .collect();
    let keys: Vec<char> = frames::columns()
        .iter()
        .filter_map(|column| column.sort_key)
        .collect();
    assert_eq!(keys.len(), 6, "six of the nine columns sort");
    for (at, key) in keys.iter().enumerate() {
        let lower: Vec<char> = key.to_lowercase().collect();
        assert!(
            !lower.iter().any(|k| bound.contains(k)),
            "column {at} sorts on `{key}`, which the screen has already bound"
        );
        assert!(
            !keys[..at].contains(key),
            "`{key}` is offered on two columns: {keys:?}"
        );
    }
}

/// Every sortable column can show its `↓` without clipping the column beside it.
///
/// Written against the drawn header rather than against the widths, because the mark may borrow
/// the gap to its right ([`crate::panes::cell::sort::grown`]) and whether it had to is exactly
/// what a width calculation would not tell you.
#[test]
fn every_sortable_column_shows_its_mark_without_clipping_the_column_beside_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let titles: Vec<&'static str> = frames::columns()
        .iter()
        .map(|column| column.title)
        .collect();
    let sortable: Vec<usize> = frames::columns()
        .iter()
        .enumerate()
        .filter_map(|(at, column)| column.sort_key.map(|_| at))
        .collect();

    for column in sortable {
        for direction in [Direction::Asc, Direction::Desc] {
            let buf = render(
                view(QueueState {
                    sort: Some(Sort { column, direction }),
                    ..QueueState::default()
                }),
                WIDE,
                TALL,
            );
            let header = line(&buf, header_row());
            assert!(
                header.contains(direction.glyph()),
                "column {column} ({}) sorted {direction:?} shows no mark: {header:?}",
                titles[column]
            );
            assert_eq!(
                header.matches(direction.glyph()).count(),
                1,
                "one column is sorted, not several: {header:?}"
            );
            for title in &titles {
                assert!(
                    header.contains(title),
                    "sorting on {} clipped the {title:?} header: {header:?}",
                    titles[column]
                );
            }
        }
    }
}

/// `Size` orders by bytes and `Age` by seconds — never by the text either of them prints.
///
/// The expected order is built here from the fixture with the comparator spelled out, rather
/// than by sorting the rendered rows, which would compare the renderer against itself and agree
/// with any order at all.
#[test]
fn size_orders_by_bytes_and_age_orders_by_seconds() {
    fn shown(column: usize, direction: Direction, field: usize) -> Vec<String> {
        frames::pane(&QueueState {
            sort: Some(Sort { column, direction }),
            ..QueueState::default()
        })
        .rows()
        .iter()
        .map(|row| match &row[field] {
            Cell::Measured { shown, .. } => shown.clone(),
            _ => panic!("Size and Age are measured values"),
        })
        .collect()
    }

    // Descending by size: the largest file in the captured page leads, and `4.0 MB` is above
    // every `KB` — which a comparison of the printed strings would file under `9`.
    let sizes = shown(frames::SIZE, Direction::Desc, frames::SIZE);
    let widest = fixture::ROWS
        .iter()
        .map(|row| row.bytes)
        .max()
        .expect("rows");
    let expected = fixture::ROWS
        .iter()
        .find(|row| row.bytes == widest)
        .expect("rows")
        .size;
    assert_eq!(
        sizes[0],
        expected,
        "the biggest file is at the top: {:?}",
        &sizes[..4]
    );
    assert!(
        sizes.iter().any(|size| size.ends_with("MB")),
        "the fixture has no megabyte row, so this proves nothing"
    );
    assert!(
        !sizes[1].ends_with("MB") || sizes[0].ends_with("MB"),
        "megabytes must not be interleaved with kilobytes: {:?}",
        &sizes[..6]
    );

    // Ascending by age: the newest row first, and `19h ago` last — where a string comparison
    // would put `19h ago` above `1m ago`.
    let ages = shown(frames::AGE, Direction::Asc, frames::AGE);
    assert_eq!(ages[0], "1m ago", "{:?}", &ages[..3]);
    assert_eq!(
        ages[ages.len() - 1],
        "19h ago",
        "{:?}",
        &ages[ages.len() - 3..]
    );
}
