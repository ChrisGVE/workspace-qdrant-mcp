with Ada.Characters.Handling; use Ada.Characters.Handling;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Containers; use Ada.Containers;
with Models; use Models;

package body Storage is

   procedure Add_Book (S : in out Shelf; B : Book) is
   begin
      if Is_Full (S) then
         raise Constraint_Error with "Shelf is at capacity";
      end if;
      S.Books.Append (B);
   end Add_Book;

   procedure Remove_Book (S : in out Shelf; ISBN : String) is
      Target : constant Unbounded_String := To_Unbounded_String (ISBN);
   begin
      for I in S.Books.First_Index .. S.Books.Last_Index loop
         if S.Books (I).ISBN = Target then
            S.Books.Delete (I);
            return;
         end if;
      end loop;
      raise Constraint_Error with "Book not found";
   end Remove_Book;

   function Find_By_Author (S : Shelf; Author : String)
      return Book_Vectors.Vector
   is
      Result : Book_Vectors.Vector;
      Query  : constant String := To_Lower (Author);
   begin
      for B of S.Books loop
         if Ada.Strings.Unbounded.Index
           (To_Unbounded_String (To_Lower (To_String (B.Author))),
            Query) > 0
         then
            Result.Append (B);
         end if;
      end loop;
      return Result;
   end Find_By_Author;

   function Find_By_Year_Range (S : Shelf; Start_Year, End_Year : Integer)
      return Book_Vectors.Vector
   is
      Result : Book_Vectors.Vector;
   begin
      for B of S.Books loop
         if B.Year >= Start_Year and then B.Year <= End_Year then
            Result.Append (B);
         end if;
      end loop;
      return Result;
   end Find_By_Year_Range;

   function Sort_By_Title (S : Shelf) return Book_Vectors.Vector is
      package Sorting is new Book_Vectors.Generic_Sorting
        ("<" => (function (L, R : Book) return Boolean is
                   (L.Title < R.Title)));
      Result : Book_Vectors.Vector := S.Books;
   begin
      Sorting.Sort (Result);
      return Result;
   end Sort_By_Title;

   function Is_Full (S : Shelf) return Boolean is
   begin
      return Integer (S.Books.Length) >= S.Capacity;
   end Is_Full;

   function Int_Image (N : Integer) return String is
      S : constant String := Integer'Image (N);
   begin
      if S (S'First) = ' ' then
         return S (S'First + 1 .. S'Last);
      end if;
      return S;
   end Int_Image;

   function Generate_Report (S : Shelf) return String is
      Total     : constant Integer := Integer (S.Books.Length);
      Available : Integer := 0;
      Min_Year  : Integer := 9999;
      Max_Year  : Integer := 0;
      Result    : Unbounded_String := Null_Unbounded_String;

      type Author_Entry is record
         Name  : Unbounded_String;
         Count : Integer;
      end record;
      type Author_Array is array (1 .. 100) of Author_Entry;
      Authors     : Author_Array;
      Num_Authors : Integer := 0;

      procedure NL is
      begin
         Append (Result, ASCII.LF);
      end NL;

      procedure Add_Author (Name : Unbounded_String) is
      begin
         for I in 1 .. Num_Authors loop
            if Authors (I).Name = Name then
               Authors (I).Count := Authors (I).Count + 1;
               return;
            end if;
         end loop;
         Num_Authors := Num_Authors + 1;
         Authors (Num_Authors) := (Name => Name, Count => 1);
      end Add_Author;

      Avail_Pct : Integer;
      Cap_Pct   : Integer;
   begin
      for B of S.Books loop
         if B.Available then
            Available := Available + 1;
         end if;
         if B.Year < Min_Year then
            Min_Year := B.Year;
         end if;
         if B.Year > Max_Year then
            Max_Year := B.Year;
         end if;
         Add_Author (B.Author);
      end loop;

      if Total > 0 then
         Avail_Pct := Available * 100 / Total;
      else
         Avail_Pct := 0;
      end if;
      if S.Capacity > 0 then
         Cap_Pct := Total * 100 / S.Capacity;
      else
         Cap_Pct := 0;
      end if;

      Append (Result, "=== Library Report: " & To_String (S.Name) & " ===");
      NL;
      Append (Result, "Total books: " & Int_Image (Total));
      NL;
      Append (Result, "Available: " & Int_Image (Available) & " / " &
              Int_Image (Total) & " (" & Int_Image (Avail_Pct) & "%)");
      NL;
      Append (Result, "Capacity: " & Int_Image (Total) & " / " &
              Int_Image (S.Capacity) & " (" & Int_Image (Cap_Pct) &
              "% full)");
      NL;
      NL;
      Append (Result, "Authors (" & Int_Image (Num_Authors) & " unique):");
      NL;
      for I in 1 .. Num_Authors loop
         Append (Result, "  - " & To_String (Authors (I).Name) & " (" &
                 Int_Image (Authors (I).Count) & " books)");
         NL;
      end loop;
      NL;
      Append (Result, "Year range: " & Int_Image (Min_Year) & " - " &
              Int_Image (Max_Year));
      NL;
      NL;
      Append (Result, "Books by availability:");
      NL;
      for B of S.Books loop
         if B.Available then
            Append (Result, "  [+] ");
         else
            Append (Result, "  [-] ");
         end if;
         Append (Result, To_String (B.Title) & " by " &
                 To_String (B.Author) & " (" & Int_Image (B.Year) &
                 ") - " & To_String (B.ISBN));
         NL;
      end loop;

      return To_String (Result);
   end Generate_Report;

end Storage;
