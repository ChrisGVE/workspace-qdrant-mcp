package Models;

use strict;
use warnings;

# Genre constants
use constant {
    GENRE_FICTION     => "Fiction",
    GENRE_NON_FICTION => "NonFiction",
    GENRE_SCIENCE     => "Science",
    GENRE_HISTORY     => "History",
    GENRE_BIOGRAPHY   => "Biography",
    GENRE_TECHNOLOGY  => "Technology",
    GENRE_PHILOSOPHY  => "Philosophy",
};

sub new_book {
    my (%args) = @_;
    return {
        title     => $args{title},
        author    => $args{author},
        year      => $args{year},
        isbn      => $args{isbn},
        available => $args{available},
    };
}

sub new_shelf {
    my (%args) = @_;
    return {
        name     => $args{name},
        capacity => $args{capacity},
        books    => [],
    };
}

1;
