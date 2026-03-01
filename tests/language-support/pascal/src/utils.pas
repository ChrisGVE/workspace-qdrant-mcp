unit Utils;

{$mode objfpc}{$H+}

interface

uses SysUtils, Models;

function ValidateISBN(const ISBN: String): Boolean;
function FormatBook(const B: TBook): String;
function ParseCSVLine(const Line: String): TBook;

implementation

function ValidateISBN(const ISBN: String): Boolean;
var
    I, D, Total: Integer;
begin
    Result := False;
    if Length(ISBN) <> 13 then Exit;
    Total := 0;
    for I := 1 to 13 do
    begin
        if (ISBN[I] < '0') or (ISBN[I] > '9') then Exit;
        D := Ord(ISBN[I]) - Ord('0');
        if ((I - 1) mod 2) = 0 then
            Total := Total + D
        else
            Total := Total + D * 3;
    end;
    Result := (Total mod 10) = 0;
end;

function FormatBook(const B: TBook): String;
begin
    Result := '"' + B.Title + '" by ' + B.Author +
              ' (' + IntToStr(B.Year) + ') [ISBN: ' + B.ISBN + ']';
end;

function ParseCSVLine(const Line: String): TBook;
var
    Parts: array of String;
    I, Start, Count: Integer;
    YearVal: Integer;
    AvailStr: String;
begin
    Count := 0;
    SetLength(Parts, 5);
    Start := 1;
    for I := 1 to Length(Line) do
    begin
        if Line[I] = ',' then
        begin
            if Count < 5 then
                Parts[Count] := Copy(Line, Start, I - Start);
            Inc(Count);
            Start := I + 1;
        end;
    end;
    if Count < 5 then
        Parts[Count] := Copy(Line, Start, Length(Line) - Start + 1);
    Inc(Count);

    if Count <> 5 then
        raise Exception.Create('Expected 5 fields');

    YearVal := StrToInt(Trim(Parts[2]));
    AvailStr := LowerCase(Trim(Parts[4]));

    if AvailStr = 'true' then
        Result := CreateBook(Trim(Parts[0]), Trim(Parts[1]),
                             YearVal, Trim(Parts[3]), True)
    else if AvailStr = 'false' then
        Result := CreateBook(Trim(Parts[0]), Trim(Parts[1]),
                             YearVal, Trim(Parts[3]), False)
    else
        raise Exception.Create('Invalid boolean: ' + Parts[4]);
end;

end.
