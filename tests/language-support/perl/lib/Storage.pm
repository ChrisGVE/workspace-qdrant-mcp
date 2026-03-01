package Storage;

use strict;
use warnings;

sub add_book {
    my ($shelf, $book) = @_;
    if (is_full($shelf)) {
        die "Shelf '$shelf->{name}' is at capacity ($shelf->{capacity})\n";
    }
    push @{$shelf->{books}}, $book;
}

sub remove_book {
    my ($shelf, $isbn) = @_;
    my $books = $shelf->{books};
    for my $i (0 .. $#$books) {
        if ($books->[$i]{isbn} eq $isbn) {
            return splice(@$books, $i, 1);
        }
    }
    die "No book with ISBN '$isbn' found on shelf\n";
}

sub find_by_author {
    my ($shelf, $author) = @_;
    my $query = lc($author);
    return grep { index(lc($_->{author}), $query) >= 0 } @{$shelf->{books}};
}

sub find_by_year_range {
    my ($shelf, $start_year, $end_year) = @_;
    return grep {
        $_->{year} >= $start_year && $_->{year} <= $end_year
    } @{$shelf->{books}};
}

sub sort_by_title {
    my ($shelf) = @_;
    return sort { $a->{title} cmp $b->{title} } @{$shelf->{books}};
}

sub is_full {
    my ($shelf) = @_;
    return scalar(@{$shelf->{books}}) >= $shelf->{capacity};
}

sub generate_report {
    my ($shelf) = @_;
    my @books = @{$shelf->{books}};
    my $total = scalar @books;
    my $available_count = scalar grep { $_->{available} } @books;
    my $avail_pct = $total > 0 ? int($available_count * 100 / $total) : 0;
    my $cap_pct = $shelf->{capacity} > 0 ? int($total * 100 / $shelf->{capacity}) : 0;

    my %author_counts;
    my @author_order;
    for my $book (@books) {
        if (!exists $author_counts{$book->{author}}) {
            push @author_order, $book->{author};
        }
        $author_counts{$book->{author}}++;
    }
    my $unique_authors = scalar @author_order;

    my @years = map { $_->{year} } @books;
    my ($min_year, $max_year) = (0, 0);
    if (@years) {
        ($min_year, $max_year) = ($years[0], $years[0]);
        for my $y (@years) {
            $min_year = $y if $y < $min_year;
            $max_year = $y if $y > $max_year;
        }
    }

    my @lines;
    push @lines, "=== Library Report: $shelf->{name} ===";
    push @lines, "Total books: $total";
    push @lines, "Available: $available_count / $total ($avail_pct%)";
    push @lines, "Capacity: $total / $shelf->{capacity} ($cap_pct% full)";
    push @lines, "";

    push @lines, "Authors ($unique_authors unique):";
    for my $author (@author_order) {
        push @lines, "  - $author ($author_counts{$author} books)";
    }
    push @lines, "";

    push @lines, "Year range: $min_year - $max_year";
    push @lines, "";

    push @lines, "Books by availability:";
    for my $book (@books) {
        my $marker = $book->{available} ? "+" : "-";
        push @lines, "  [$marker] $book->{title} by $book->{author} ($book->{year}) - $book->{isbn}";
    }

    return join("\n", @lines);
}

1;
