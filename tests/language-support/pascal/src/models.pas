unit Models;

{$mode objfpc}{$H+}

interface

uses SysUtils;

type
    TGenre = (gFiction, gNonFiction, gScience, gHistory,
              gBiography, gTechnology, gPhilosophy);

    TBook = record
        Title: String;
        Author: String;
        Year: Integer;
        ISBN: String;
        Available: Boolean;
    end;

    TBookArray = array of TBook;

    TShelf = record
        Name: String;
        Capacity: Integer;
        Books: TBookArray;
    end;

function CreateBook(const Title, Author: String; Year: Integer;
                    const ISBN: String; Available: Boolean): TBook;
function CreateShelf(const Name: String; Capacity: Integer): TShelf;

implementation

function CreateBook(const Title, Author: String; Year: Integer;
                    const ISBN: String; Available: Boolean): TBook;
begin
    Result.Title := Title;
    Result.Author := Author;
    Result.Year := Year;
    Result.ISBN := ISBN;
    Result.Available := Available;
end;

function CreateShelf(const Name: String; Capacity: Integer): TShelf;
begin
    Result.Name := Name;
    Result.Capacity := Capacity;
    SetLength(Result.Books, 0);
end;

end.
