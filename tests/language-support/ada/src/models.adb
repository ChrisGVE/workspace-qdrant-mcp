package body Models is

   function Create_Book (Title     : String;
                         Author    : String;
                         Year      : Integer;
                         ISBN      : String;
                         Available : Boolean) return Book is
   begin
      return (Title     => To_Unbounded_String (Title),
              Author    => To_Unbounded_String (Author),
              Year      => Year,
              ISBN      => To_Unbounded_String (ISBN),
              Available => Available);
   end Create_Book;

   function Create_Shelf (Name     : String;
                          Capacity : Integer) return Shelf is
   begin
      return (Name     => To_Unbounded_String (Name),
              Capacity => Capacity,
              Books    => Book_Vectors.Empty_Vector);
   end Create_Shelf;

end Models;
