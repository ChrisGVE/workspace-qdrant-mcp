module utils_mod
    use models_mod
    implicit none

contains

    function validate_isbn(isbn) result(valid)
        character(len=*), intent(in) :: isbn
        logical :: valid
        integer :: i, total, d, w
        valid = .false.
        if (len_trim(isbn) /= 13) return
        total = 0
        do i = 1, 13
            d = ichar(isbn(i:i)) - ichar('0')
            if (d < 0 .or. d > 9) return
            if (mod(i - 1, 2) == 0) then
                w = 1
            else
                w = 3
            end if
            total = total + d * w
        end do
        valid = (mod(total, 10) == 0)
    end function validate_isbn

    function format_book(b) result(s)
        type(Book), intent(in) :: b
        character(len=512) :: s
        character(len=20) :: year_str
        write(year_str, '(I0)') b%year
        s = '"' // trim(b%title) // '" by ' // trim(b%author) // &
            ' (' // trim(year_str) // ') [ISBN: ' // trim(b%isbn) // ']'
    end function format_book

    function parse_csv_line(line) result(b)
        character(len=*), intent(in) :: line
        type(Book) :: b
        integer :: p1, p2, p3, p4, ios
        character(len=256) :: title_s, author_s, year_s, isbn_s, avail_s

        p1 = index(line, ',')
        title_s = line(1:p1-1)
        p2 = index(line(p1+1:), ',') + p1
        author_s = line(p1+1:p2-1)
        p3 = index(line(p2+1:), ',') + p2
        year_s = line(p2+1:p3-1)
        p4 = index(line(p3+1:), ',') + p3
        isbn_s = line(p3+1:p4-1)
        avail_s = line(p4+1:)

        b%title = trim(title_s)
        b%author = trim(author_s)
        read(year_s, *, iostat=ios) b%year
        b%isbn = trim(isbn_s)
        if (trim(adjustl(avail_s)) == 'true') then
            b%available = .true.
        else
            b%available = .false.
        end if
    end function parse_csv_line

end module utils_mod
