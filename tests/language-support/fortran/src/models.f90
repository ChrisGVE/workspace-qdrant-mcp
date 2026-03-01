module models_mod
    implicit none

    integer, parameter :: GENRE_FICTION = 1
    integer, parameter :: GENRE_NON_FICTION = 2
    integer, parameter :: GENRE_SCIENCE = 3
    integer, parameter :: GENRE_HISTORY = 4
    integer, parameter :: GENRE_BIOGRAPHY = 5
    integer, parameter :: GENRE_TECHNOLOGY = 6
    integer, parameter :: GENRE_PHILOSOPHY = 7

    type :: Book
        character(len=256) :: title
        character(len=128) :: author
        integer :: year
        character(len=13) :: isbn
        logical :: available
    end type Book

    type :: Shelf
        character(len=128) :: name
        integer :: capacity
        type(Book), allocatable :: books(:)
        integer :: count
    end type Shelf

contains

    function create_book(title, author, year, isbn, available) result(b)
        character(len=*), intent(in) :: title, author, isbn
        integer, intent(in) :: year
        logical, intent(in) :: available
        type(Book) :: b
        b%title = title
        b%author = author
        b%year = year
        b%isbn = isbn
        b%available = available
    end function create_book

    function create_shelf(name, capacity) result(s)
        character(len=*), intent(in) :: name
        integer, intent(in) :: capacity
        type(Shelf) :: s
        s%name = name
        s%capacity = capacity
        s%count = 0
        allocate(s%books(capacity))
    end function create_shelf

end module models_mod
