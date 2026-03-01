program Bookshelf;

{$mode objfpc}{$H+}

uses SysUtils, Models, Storage, Utils;

var
    S: TShelf;
    Results: TBookArray;
    Parsed: TBook;
    I: Integer;
    Report: String;
begin
    S := CreateShelf('Computer Science', 10);

    AddBook(S, CreateBook('The Art of Computer Programming',
                          'Donald Knuth', 1968, '9780201896831', True));
    AddBook(S, CreateBook('Structure and Interpretation of Computer Programs',
                          'Harold Abelson', 1996, '9780262510875', True));
    AddBook(S, CreateBook('Introduction to Algorithms',
                          'Thomas Cormen', 2009, '9780262033848', False));
    AddBook(S, CreateBook('Design Patterns',
                          'Erich Gamma', 1994, '9780201633610', True));
    AddBook(S, CreateBook('The Pragmatic Programmer',
                          'David Thomas', 2019, '9780135957059', True));

    Report := GenerateReport(S);
    Write(Report);
    WriteLn;

    WriteLn('--- Search by author "knuth" ---');
    Results := FindByAuthor(S, 'knuth');
    for I := 0 to Length(Results) - 1 do
        WriteLn('  ', FormatBook(Results[I]));
    WriteLn;

    WriteLn('--- Search by year range 1990-2010 ---');
    Results := FindByYearRange(S, 1990, 2010);
    for I := 0 to Length(Results) - 1 do
        WriteLn('  ', FormatBook(Results[I]));
    WriteLn;

    WriteLn('--- Parse CSV ---');
    Parsed := ParseCSVLine('Clean Code,Robert Martin,2008,9780132350884,true');
    WriteLn('  Parsed: ', FormatBook(Parsed));
    WriteLn;

    WriteLn('--- ISBN Validation ---');
    for I := 0 to Length(S.Books) - 1 do
    begin
        if ValidateISBN(S.Books[I].ISBN) then
            WriteLn('  ', S.Books[I].ISBN, ': valid')
        else
            WriteLn('  ', S.Books[I].ISBN, ': invalid');
    end;
end.
