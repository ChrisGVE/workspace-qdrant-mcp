module storage_mod
    use models_mod
    implicit none

contains

    subroutine add_book(s, b)
        type(Shelf), intent(inout) :: s
        type(Book), intent(in) :: b
        if (is_full(s)) then
            print *, "Error: shelf at capacity"
            stop 1
        end if
        s%count = s%count + 1
        s%books(s%count) = b
    end subroutine add_book

    function is_full(s) result(full)
        type(Shelf), intent(in) :: s
        logical :: full
        full = (s%count >= s%capacity)
    end function is_full

    function find_by_author(s, author, results) result(n)
        type(Shelf), intent(in) :: s
        character(len=*), intent(in) :: author
        type(Book), intent(out) :: results(:)
        integer :: n, i
        character(len=256) :: lower_author, lower_book_author
        n = 0
        lower_author = to_lower(trim(author))
        do i = 1, s%count
            lower_book_author = to_lower(trim(s%books(i)%author))
            if (index(lower_book_author, trim(lower_author)) > 0) then
                n = n + 1
                results(n) = s%books(i)
            end if
        end do
    end function find_by_author

    function find_by_year_range(s, start_year, end_year, results) result(n)
        type(Shelf), intent(in) :: s
        integer, intent(in) :: start_year, end_year
        type(Book), intent(out) :: results(:)
        integer :: n, i
        n = 0
        do i = 1, s%count
            if (s%books(i)%year >= start_year .and. &
                s%books(i)%year <= end_year) then
                n = n + 1
                results(n) = s%books(i)
            end if
        end do
    end function find_by_year_range

    function to_lower(str) result(lower)
        character(len=*), intent(in) :: str
        character(len=256) :: lower
        integer :: i, ic
        lower = str
        do i = 1, len_trim(str)
            ic = ichar(str(i:i))
            if (ic >= ichar('A') .and. ic <= ichar('Z')) then
                lower(i:i) = char(ic + 32)
            end if
        end do
    end function to_lower

    subroutine append_buf(buf, pos, str)
        character(len=*), intent(inout) :: buf
        integer, intent(inout) :: pos
        character(len=*), intent(in) :: str
        integer :: slen
        slen = len(str)
        buf(pos+1:pos+slen) = str
        pos = pos + slen
    end subroutine append_buf

    subroutine append_int(buf, pos, n)
        character(len=*), intent(inout) :: buf
        integer, intent(inout) :: pos
        integer, intent(in) :: n
        character(len=20) :: tmp
        write(tmp, '(I0)') n
        call append_buf(buf, pos, trim(tmp))
    end subroutine append_int

    subroutine append_nl(buf, pos)
        character(len=*), intent(inout) :: buf
        integer, intent(inout) :: pos
        call append_buf(buf, pos, char(10))
    end subroutine append_nl

    subroutine generate_report(s, report)
        type(Shelf), intent(in) :: s
        character(len=*), intent(out) :: report
        integer :: i, j, total, avail_count, avail_pct, cap_pct
        integer :: min_year, max_year, num_authors, pos
        character(len=128) :: author_names(100)
        integer :: author_counts(100)
        logical :: found

        total = s%count
        avail_count = 0
        min_year = 9999
        max_year = 0
        num_authors = 0

        do i = 1, total
            if (s%books(i)%available) avail_count = avail_count + 1
            if (s%books(i)%year < min_year) min_year = s%books(i)%year
            if (s%books(i)%year > max_year) max_year = s%books(i)%year
            found = .false.
            do j = 1, num_authors
                if (trim(author_names(j)) == trim(s%books(i)%author)) then
                    author_counts(j) = author_counts(j) + 1
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                num_authors = num_authors + 1
                author_names(num_authors) = s%books(i)%author
                author_counts(num_authors) = 1
            end if
        end do

        if (total > 0) then
            avail_pct = avail_count * 100 / total
        else
            avail_pct = 0
        end if
        if (s%capacity > 0) then
            cap_pct = total * 100 / s%capacity
        else
            cap_pct = 0
        end if

        report = ''
        pos = 0
        call append_buf(report, pos, '=== Library Report: ')
        call append_buf(report, pos, trim(s%name))
        call append_buf(report, pos, ' ===')
        call append_nl(report, pos)
        call append_buf(report, pos, 'Total books: ')
        call append_int(report, pos, total)
        call append_nl(report, pos)
        call append_buf(report, pos, 'Available: ')
        call append_int(report, pos, avail_count)
        call append_buf(report, pos, ' / ')
        call append_int(report, pos, total)
        call append_buf(report, pos, ' (')
        call append_int(report, pos, avail_pct)
        call append_buf(report, pos, '%)')
        call append_nl(report, pos)
        call append_buf(report, pos, 'Capacity: ')
        call append_int(report, pos, total)
        call append_buf(report, pos, ' / ')
        call append_int(report, pos, s%capacity)
        call append_buf(report, pos, ' (')
        call append_int(report, pos, cap_pct)
        call append_buf(report, pos, '% full)')
        call append_nl(report, pos)
        call append_nl(report, pos)
        call append_buf(report, pos, 'Authors (')
        call append_int(report, pos, num_authors)
        call append_buf(report, pos, ' unique):')
        call append_nl(report, pos)
        do i = 1, num_authors
            call append_buf(report, pos, '  - ')
            call append_buf(report, pos, trim(author_names(i)))
            call append_buf(report, pos, ' (')
            call append_int(report, pos, author_counts(i))
            call append_buf(report, pos, ' books)')
            call append_nl(report, pos)
        end do
        call append_nl(report, pos)
        call append_buf(report, pos, 'Year range: ')
        call append_int(report, pos, min_year)
        call append_buf(report, pos, ' - ')
        call append_int(report, pos, max_year)
        call append_nl(report, pos)
        call append_nl(report, pos)
        call append_buf(report, pos, 'Books by availability:')
        call append_nl(report, pos)
        do i = 1, total
            if (s%books(i)%available) then
                call append_buf(report, pos, '  [+] ')
            else
                call append_buf(report, pos, '  [-] ')
            end if
            call append_buf(report, pos, trim(s%books(i)%title))
            call append_buf(report, pos, ' by ')
            call append_buf(report, pos, trim(s%books(i)%author))
            call append_buf(report, pos, ' (')
            call append_int(report, pos, s%books(i)%year)
            call append_buf(report, pos, ') - ')
            call append_buf(report, pos, trim(s%books(i)%isbn))
            call append_nl(report, pos)
        end do
    end subroutine generate_report

end module storage_mod
