with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Models; use Models;
with Storage; use Storage;
with Utils; use Utils;

procedure Main is
   S : Shelf := Create_Shelf ("Computer Science", 10);

   Books : constant array (1 .. 5) of Book :=
     (Create_Book ("The Art of Computer Programming",
                   "Donald Knuth", 1968, "9780201896831", True),
      Create_Book ("Structure and Interpretation of Computer Programs",
                   "Harold Abelson", 1996, "9780262510875", True),
      Create_Book ("Introduction to Algorithms",
                   "Thomas Cormen", 2009, "9780262033848", False),
      Create_Book ("Design Patterns",
                   "Erich Gamma", 1994, "9780201633610", True),
      Create_Book ("The Pragmatic Programmer",
                   "David Thomas", 2019, "9780135957059", True));

   Results : Book_Vectors.Vector;
begin
   for B of Books loop
      Add_Book (S, B);
   end loop;

   Put (Generate_Report (S));
   New_Line;

   Put_Line ("--- Search by author ""knuth"" ---");
   Results := Find_By_Author (S, "knuth");
   for B of Results loop
      Put_Line ("  " & Format_Book (B));
   end loop;
   New_Line;

   Put_Line ("--- Search by year range 1990-2010 ---");
   Results := Find_By_Year_Range (S, 1990, 2010);
   for B of Results loop
      Put_Line ("  " & Format_Book (B));
   end loop;
   New_Line;

   Put_Line ("--- Parse CSV ---");
   declare
      Parsed : constant Book :=
        Parse_CSV_Line ("Clean Code,Robert Martin,2008,9780132350884,true");
   begin
      Put_Line ("  Parsed: " & Format_Book (Parsed));
   end;
   New_Line;

   Put_Line ("--- ISBN Validation ---");
   for B of Books loop
      if Validate_ISBN (To_String (B.ISBN)) then
         Put_Line ("  " & To_String (B.ISBN) & ": valid");
      else
         Put_Line ("  " & To_String (B.ISBN) & ": invalid");
      end if;
   end loop;
end Main;
