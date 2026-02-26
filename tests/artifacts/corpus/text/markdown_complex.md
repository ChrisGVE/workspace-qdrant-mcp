# Complex Markdown Document

## Introduction

This document tests **bold**, *italic*, `inline code`, and ~~strikethrough~~ formatting.

## Code Blocks

```rust
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
```

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

## Tables

| Column A | Column B | Column C |
|----------|----------|----------|
| 1        | Alpha    | true     |
| 2        | Beta     | false    |
| 3        | Gamma    | true     |

## Lists

1. First ordered item
   - Nested unordered
   - Another nested
2. Second ordered item
3. Third ordered item

## Links and Images

[Example Link](https://example.com)

> This is a blockquote with **bold** text inside.
> It spans multiple lines.

---

## Math (LaTeX)

Inline: $E = mc^2$

Block:
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

## Footnotes

This has a footnote[^1].

[^1]: This is the footnote content.
