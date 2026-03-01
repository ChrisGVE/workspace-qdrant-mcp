program bookshelf
    use models_mod
    use storage_mod
    use utils_mod
    implicit none

    type(Shelf) :: s
    type(Book) :: results(10), parsed
    integer :: n, i
    character(len=4096) :: report

    s = create_shelf("Computer Science", 10)

    call add_book(s, create_book( &
        "The Art of Computer Programming", &
        "Donald Knuth", 1968, "9780201896831", .true.))
    call add_book(s, create_book( &
        "Structure and Interpretation of Computer Programs", &
        "Harold Abelson", 1996, "9780262510875", .true.))
    call add_book(s, create_book( &
        "Introduction to Algorithms", &
        "Thomas Cormen", 2009, "9780262033848", .false.))
    call add_book(s, create_book( &
        "Design Patterns", &
        "Erich Gamma", 1994, "9780201633610", .true.))
    call add_book(s, create_book( &
        "The Pragmatic Programmer", &
        "David Thomas", 2019, "9780135957059", .true.))

    call generate_report(s, report)
    write(*, '(A)', advance='no') trim(report)
    write(*, '(A)') ''

    write(*, '(A)') '--- Search by author "knuth" ---'
    n = find_by_author(s, "knuth", results)
    do i = 1, n
        write(*, '(A,A)') '  ', trim(format_book(results(i)))
    end do
    write(*, '(A)') ''

    write(*, '(A)') '--- Search by year range 1990-2010 ---'
    n = find_by_year_range(s, 1990, 2010, results)
    do i = 1, n
        write(*, '(A,A)') '  ', trim(format_book(results(i)))
    end do
    write(*, '(A)') ''

    write(*, '(A)') '--- Parse CSV ---'
    parsed = parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true")
    write(*, '(A,A)') '  Parsed: ', trim(format_book(parsed))
    write(*, '(A)') ''

    write(*, '(A)') '--- ISBN Validation ---'
    do i = 1, s%count
        if (validate_isbn(s%books(i)%isbn)) then
            write(*, '(A,A,A)') '  ', trim(s%books(i)%isbn), ': valid'
        else
            write(*, '(A,A,A)') '  ', trim(s%books(i)%isbn), ': invalid'
        end if
    end do

end program bookshelf
