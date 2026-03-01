with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Characters.Handling; use Ada.Characters.Handling;
with Models; use Models;

package body Utils is

   function Validate_ISBN (ISBN : String) return Boolean is
      Total : Integer := 0;
   begin
      if ISBN'Length /= 13 then
         return False;
      end if;
      for I in ISBN'Range loop
         if not Is_Digit (ISBN (I)) then
            return False;
         end if;
         declare
            D   : constant Integer :=
              Character'Pos (ISBN (I)) - Character'Pos ('0');
            Idx : constant Integer := I - ISBN'First;
         begin
            if Idx mod 2 = 0 then
               Total := Total + D;
            else
               Total := Total + D * 3;
            end if;
         end;
      end loop;
      return Total mod 10 = 0;
   end Validate_ISBN;

   function Int_Image (N : Integer) return String is
      S : constant String := Integer'Image (N);
   begin
      if S (S'First) = ' ' then
         return S (S'First + 1 .. S'Last);
      end if;
      return S;
   end Int_Image;

   function Format_Book (B : Book) return String is
   begin
      return """" & To_String (B.Title) & """ by " &
             To_String (B.Author) & " (" & Int_Image (B.Year) &
             ") [ISBN: " & To_String (B.ISBN) & "]";
   end Format_Book;

   function Parse_CSV_Line (Line : String) return Book is
      Fields : array (1 .. 5) of Unbounded_String;
      Idx    : Integer := 1;
      Start  : Integer := Line'First;
   begin
      for I in Line'Range loop
         if Line (I) = ',' then
            Fields (Idx) := To_Unbounded_String (Line (Start .. I - 1));
            Idx := Idx + 1;
            Start := I + 1;
         end if;
      end loop;
      Fields (Idx) := To_Unbounded_String (Line (Start .. Line'Last));

      if Idx /= 5 then
         raise Constraint_Error with "Expected 5 fields";
      end if;

      declare
         Year_Str : constant String := To_String (Fields (3));
         Avail_Str : constant String :=
           To_Lower (To_String (Fields (5)));
         Year : Integer;
         Avail : Boolean;
      begin
         Year := Integer'Value (Year_Str);
         if Avail_Str = "true" then
            Avail := True;
         elsif Avail_Str = "false" then
            Avail := False;
         else
            raise Constraint_Error with "Invalid boolean";
         end if;
         return Create_Book (To_String (Fields (1)),
                             To_String (Fields (2)),
                             Year,
                             To_String (Fields (4)),
                             Avail);
      end;
   end Parse_CSV_Line;

end Utils;
