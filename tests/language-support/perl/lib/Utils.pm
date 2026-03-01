package Utils;

use strict;
use warnings;
use Models;

sub validate_isbn {
    my ($isbn) = @_;
    return 0 unless defined $isbn && length($isbn) == 13 && $isbn =~ /^\d{13}$/;
    my $total = 0;
    for my $i (0 .. 12) {
        my $digit = substr($isbn, $i, 1);
        $total += $digit * ($i % 2 == 0 ? 1 : 3);
    }
    return ($total % 10 == 0) ? 1 : 0;
}

sub format_book {
    my ($book) = @_;
    return "\"$book->{title}\" by $book->{author} ($book->{year}) [ISBN: $book->{isbn}]";
}

sub parse_csv_line {
    my ($line) = @_;
    my @parts = split(/,/, $line);
    die "Expected 5 fields, got " . scalar(@parts) . "\n" unless @parts == 5;

    my ($title, $author, $year_str, $isbn, $avail_str) = @parts;
    $title =~ s/^\s+|\s+$//g;
    $author =~ s/^\s+|\s+$//g;
    $isbn =~ s/^\s+|\s+$//g;
    $avail_str =~ s/^\s+|\s+$//g;

    die "Invalid year: $year_str\n" unless $year_str =~ /^\d+$/;
    my $year = int($year_str);

    my $available;
    if (lc($avail_str) eq 'true') {
        $available = 1;
    } elsif (lc($avail_str) eq 'false') {
        $available = 0;
    } else {
        die "Invalid boolean: $avail_str\n";
    }

    return Models::new_book(
        title     => $title,
        author    => $author,
        year      => $year,
        isbn      => $isbn,
        available => $available,
    );
}

1;
