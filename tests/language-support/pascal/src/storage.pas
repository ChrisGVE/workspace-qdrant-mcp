unit Storage;

{$mode objfpc}{$H+}

interface

uses SysUtils, Models;

procedure AddBook(var S: TShelf; const B: TBook);
procedure RemoveBook(var S: TShelf; const ISBN: String);
function FindByAuthor(const S: TShelf; const Author: String): TBookArray;
function FindByYearRange(const S: TShelf;
                         StartYear, EndYear: Integer): TBookArray;
function SortByTitle(const S: TShelf): TBookArray;
function IsFull(const S: TShelf): Boolean;
function GenerateReport(const S: TShelf): String;

implementation

procedure AddBook(var S: TShelf; const B: TBook);
var
    N: Integer;
begin
    if IsFull(S) then
        raise Exception.Create('Shelf is at capacity');
    N := Length(S.Books);
    SetLength(S.Books, N + 1);
    S.Books[N] := B;
end;

procedure RemoveBook(var S: TShelf; const ISBN: String);
var
    I, J, N: Integer;
begin
    N := Length(S.Books);
    for I := 0 to N - 1 do
    begin
        if S.Books[I].ISBN = ISBN then
        begin
            for J := I to N - 2 do
                S.Books[J] := S.Books[J + 1];
            SetLength(S.Books, N - 1);
            Exit;
        end;
    end;
    raise Exception.Create('Book not found');
end;

function FindByAuthor(const S: TShelf; const Author: String): TBookArray;
var
    I, Count: Integer;
    Query: String;
begin
    Count := 0;
    SetLength(Result, 0);
    Query := LowerCase(Author);
    for I := 0 to Length(S.Books) - 1 do
    begin
        if Pos(Query, LowerCase(S.Books[I].Author)) > 0 then
        begin
            SetLength(Result, Count + 1);
            Result[Count] := S.Books[I];
            Inc(Count);
        end;
    end;
end;

function FindByYearRange(const S: TShelf;
                         StartYear, EndYear: Integer): TBookArray;
var
    I, Count: Integer;
begin
    Count := 0;
    SetLength(Result, 0);
    for I := 0 to Length(S.Books) - 1 do
    begin
        if (S.Books[I].Year >= StartYear) and
           (S.Books[I].Year <= EndYear) then
        begin
            SetLength(Result, Count + 1);
            Result[Count] := S.Books[I];
            Inc(Count);
        end;
    end;
end;

function SortByTitle(const S: TShelf): TBookArray;
var
    I, J, N: Integer;
    Tmp: TBook;
begin
    N := Length(S.Books);
    SetLength(Result, N);
    for I := 0 to N - 1 do
        Result[I] := S.Books[I];
    for I := 0 to N - 2 do
        for J := 0 to N - 2 - I do
            if Result[J].Title > Result[J + 1].Title then
            begin
                Tmp := Result[J];
                Result[J] := Result[J + 1];
                Result[J + 1] := Tmp;
            end;
end;

function IsFull(const S: TShelf): Boolean;
begin
    Result := Length(S.Books) >= S.Capacity;
end;

function GenerateReport(const S: TShelf): String;
var
    I, J, Total, AvailCount, AvailPct, CapPct: Integer;
    MinYear, MaxYear, NumAuthors: Integer;
    AuthorNames: array[0..99] of String;
    AuthorCounts: array[0..99] of Integer;
    Found: Boolean;
    Marker: String;
begin
    Total := Length(S.Books);
    AvailCount := 0;
    MinYear := 9999;
    MaxYear := 0;
    NumAuthors := 0;

    for I := 0 to Total - 1 do
    begin
        if S.Books[I].Available then Inc(AvailCount);
        if S.Books[I].Year < MinYear then MinYear := S.Books[I].Year;
        if S.Books[I].Year > MaxYear then MaxYear := S.Books[I].Year;
        Found := False;
        for J := 0 to NumAuthors - 1 do
        begin
            if AuthorNames[J] = S.Books[I].Author then
            begin
                Inc(AuthorCounts[J]);
                Found := True;
                Break;
            end;
        end;
        if not Found then
        begin
            AuthorNames[NumAuthors] := S.Books[I].Author;
            AuthorCounts[NumAuthors] := 1;
            Inc(NumAuthors);
        end;
    end;

    if Total > 0 then
        AvailPct := AvailCount * 100 div Total
    else
        AvailPct := 0;
    if S.Capacity > 0 then
        CapPct := Total * 100 div S.Capacity
    else
        CapPct := 0;

    Result := '=== Library Report: ' + S.Name + ' ===' + LineEnding;
    Result := Result + 'Total books: ' + IntToStr(Total) + LineEnding;
    Result := Result + 'Available: ' + IntToStr(AvailCount) + ' / ' +
              IntToStr(Total) + ' (' + IntToStr(AvailPct) + '%)' +
              LineEnding;
    Result := Result + 'Capacity: ' + IntToStr(Total) + ' / ' +
              IntToStr(S.Capacity) + ' (' + IntToStr(CapPct) +
              '% full)' + LineEnding;
    Result := Result + LineEnding;
    Result := Result + 'Authors (' + IntToStr(NumAuthors) +
              ' unique):' + LineEnding;
    for I := 0 to NumAuthors - 1 do
        Result := Result + '  - ' + AuthorNames[I] + ' (' +
                  IntToStr(AuthorCounts[I]) + ' books)' + LineEnding;
    Result := Result + LineEnding;
    Result := Result + 'Year range: ' + IntToStr(MinYear) + ' - ' +
              IntToStr(MaxYear) + LineEnding;
    Result := Result + LineEnding;
    Result := Result + 'Books by availability:' + LineEnding;
    for I := 0 to Total - 1 do
    begin
        if S.Books[I].Available then
            Marker := '+'
        else
            Marker := '-';
        Result := Result + '  [' + Marker + '] ' + S.Books[I].Title +
                  ' by ' + S.Books[I].Author + ' (' +
                  IntToStr(S.Books[I].Year) + ') - ' +
                  S.Books[I].ISBN + LineEnding;
    end;
end;

end.
