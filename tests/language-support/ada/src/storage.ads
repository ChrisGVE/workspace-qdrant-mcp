with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Models; use Models;

package Storage is

   procedure Add_Book (S : in out Shelf; B : Book);
   procedure Remove_Book (S : in out Shelf; ISBN : String);
   function Find_By_Author (S : Shelf; Author : String)
      return Book_Vectors.Vector;
   function Find_By_Year_Range (S : Shelf; Start_Year, End_Year : Integer)
      return Book_Vectors.Vector;
   function Sort_By_Title (S : Shelf) return Book_Vectors.Vector;
   function Is_Full (S : Shelf) return Boolean;
   function Generate_Report (S : Shelf) return String;

end Storage;
