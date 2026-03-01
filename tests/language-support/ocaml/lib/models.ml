(** Data models for the bookshelf library management system. *)

(** Genre classification for books. *)
type genre =
  | Fiction
  | NonFiction
  | Science
  | History
  | Biography
  | Technology
  | Philosophy

(** Represents a single book in the library. *)
type book = {
  title : string;
  author : string;
  year : int;
  isbn : string;
  available : bool;
}

(** Represents a bookshelf with a fixed capacity. *)
type shelf = {
  name : string;
  capacity : int;
  mutable books : book list;
}

(** Create a new empty shelf. *)
let new_shelf name capacity = { name; capacity; books = [] }
