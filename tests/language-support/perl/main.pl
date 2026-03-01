#!/usr/bin/env perl
use strict;
use warnings;
use lib 'lib';
use Models;
use Storage;
use Utils;

sub main {
    my $shelf = Models::new_shelf(name => "Computer Science", capacity => 10);

    my @books = (
        Models::new_book(title => "The Art of Computer Programming", author => "Donald Knuth", year => 1968, isbn => "9780201896831", available => 1),
        Models::new_book(title => "Structure and Interpretation of Computer Programs", author => "Harold Abelson", year => 1996, isbn => "9780262510875", available => 1),
        Models::new_book(title => "Introduction to Algorithms", author => "Thomas Cormen", year => 2009, isbn => "9780262033848", available => 0),
        Models::new_book(title => "Design Patterns", author => "Erich Gamma", year => 1994, isbn => "9780201633610", available => 1),
        Models::new_book(title => "The Pragmatic Programmer", author => "David Thomas", year => 2019, isbn => "9780135957059", available => 1),
    );
    for my $book (@books) {
        Storage::add_book($shelf, $book);
    }

    print Storage::generate_report($shelf) . "\n";
    print "\n";

    print "--- Search by author \"knuth\" ---\n";
    for my $book (Storage::find_by_author($shelf, "knuth")) {
        print "  " . Utils::format_book($book) . "\n";
    }
    print "\n";

    print "--- Search by year range 1990-2010 ---\n";
    for my $book (Storage::find_by_year_range($shelf, 1990, 2010)) {
        print "  " . Utils::format_book($book) . "\n";
    }
    print "\n";

    print "--- Parse CSV ---\n";
    my $parsed = Utils::parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true");
    print "  Parsed: " . Utils::format_book($parsed) . "\n";
    print "\n";

    print "--- ISBN Validation ---\n";
    for my $book (@books) {
        my $status = Utils::validate_isbn($book->{isbn}) ? "valid" : "invalid";
        print "  $book->{isbn}: $status\n";
    }
}

main();
