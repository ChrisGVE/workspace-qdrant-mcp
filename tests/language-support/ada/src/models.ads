with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Containers.Vectors;

package Models is

   type Genre is (Fiction, Non_Fiction, Science, History,
                  Biography, Technology, Philosophy);

   type Book is record
      Title     : Unbounded_String;
      Author    : Unbounded_String;
      Year      : Integer;
      ISBN      : Unbounded_String;
      Available : Boolean;
   end record;

   package Book_Vectors is new Ada.Containers.Vectors
     (Index_Type   => Natural,
      Element_Type => Book);

   type Shelf is record
      Name     : Unbounded_String;
      Capacity : Integer;
      Books    : Book_Vectors.Vector;
   end record;

   function Create_Book (Title     : String;
                         Author    : String;
                         Year      : Integer;
                         ISBN      : String;
                         Available : Boolean) return Book;

   function Create_Shelf (Name     : String;
                          Capacity : Integer) return Shelf;

end Models;
