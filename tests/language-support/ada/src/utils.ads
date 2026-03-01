with Models; use Models;

package Utils is

   function Validate_ISBN (ISBN : String) return Boolean;
   function Format_Book (B : Book) return String;
   function Parse_CSV_Line (Line : String) return Book;

end Utils;
