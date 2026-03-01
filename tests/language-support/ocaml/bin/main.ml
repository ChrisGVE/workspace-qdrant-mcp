(** Bookshelf demonstration: creates a shelf, adds books, and runs queries. *)

open Models
open Storage
open Utils

let () =
  let shelf = new_shelf "Computer Science" 10 in
  let books = [
    { title = "The Art of Computer Programming"; author = "Donald Knuth"; year = 1968; isbn = "9780201896831"; available = true };
    { title = "Structure and Interpretation of Computer Programs"; author = "Harold Abelson"; year = 1996; isbn = "9780262510875"; available = true };
    { title = "Introduction to Algorithms"; author = "Thomas Cormen"; year = 2009; isbn = "9780262033848"; available = false };
    { title = "Design Patterns"; author = "Erich Gamma"; year = 1994; isbn = "9780201633610"; available = true };
    { title = "The Pragmatic Programmer"; author = "David Thomas"; year = 2019; isbn = "9780135957059"; available = true };
  ] in
  List.iter (fun b ->
    match add_book shelf b with
    | Ok () -> ()
    | Error e -> Printf.eprintf "Error: %s\n" e
  ) books;

  print_string (generate_report shelf);
  print_newline ();
  print_newline ();

  Printf.printf "--- Search by author \"knuth\" ---\n";
  List.iter (fun b ->
    Printf.printf "  %s\n" (format_book b)
  ) (find_by_author shelf "knuth");
  print_newline ();

  Printf.printf "--- Search by year range 1990-2010 ---\n";
  List.iter (fun b ->
    Printf.printf "  %s\n" (format_book b)
  ) (find_by_year_range shelf 1990 2010);
  print_newline ();

  Printf.printf "--- Parse CSV ---\n";
  (match parse_csv_line "Clean Code,Robert Martin,2008,9780132350884,true" with
   | Ok book -> Printf.printf "  Parsed: %s\n" (format_book book)
   | Error e -> Printf.printf "  Error: %s\n" e);
  print_newline ();

  Printf.printf "--- ISBN Validation ---\n";
  List.iter (fun b ->
    let status = if validate_isbn b.isbn then "valid" else "invalid" in
    Printf.printf "  %s: %s\n" b.isbn status
  ) books
