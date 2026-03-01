(** Utility functions for ISBN validation, formatting, and CSV parsing. *)

open Models

(** Validate an ISBN-13 check digit. *)
let validate_isbn isbn =
  if String.length isbn <> 13 then false
  else if not (String.to_seq isbn |> Seq.for_all (fun c -> c >= '0' && c <= '9')) then false
  else
    let total = ref 0 in
    for i = 0 to 12 do
      let d = Char.code isbn.[i] - Char.code '0' in
      let w = if i mod 2 = 0 then 1 else 3 in
      total := !total + d * w
    done;
    !total mod 10 = 0

(** Format a book as a single descriptive line. *)
let format_book book =
  Printf.sprintf "\"%s\" by %s (%d) [ISBN: %s]"
    book.title book.author book.year book.isbn

(** Split a string on a character. *)
let split_on_char sep s =
  let rec aux i acc =
    match String.index_from_opt s i sep with
    | None -> List.rev (String.sub s i (String.length s - i) :: acc)
    | Some j -> aux (j + 1) (String.sub s i (j - i) :: acc)
  in
  aux 0 []

(** Strip leading and trailing whitespace. *)
let strip s =
  let len = String.length s in
  let i = ref 0 in
  while !i < len && s.[!i] = ' ' do incr i done;
  let j = ref (len - 1) in
  while !j >= !i && s.[!j] = ' ' do decr j done;
  String.sub s !i (!j - !i + 1)

(** Parse a CSV line into a Book. *)
let parse_csv_line line =
  let parts = split_on_char ',' line in
  if List.length parts <> 5 then
    Error (Printf.sprintf "Expected 5 fields, got %d" (List.length parts))
  else
    let title = List.nth parts 0 in
    let author = List.nth parts 1 in
    let year_str = List.nth parts 2 in
    let isbn = List.nth parts 3 in
    let avail_str = List.nth parts 4 in
    match int_of_string_opt year_str with
    | None -> Error (Printf.sprintf "Invalid year: %s" year_str)
    | Some year ->
      let trimmed = String.lowercase_ascii (strip avail_str) in
      if trimmed = "true" then
        Ok { title = strip title; author = strip author; year; isbn = strip isbn; available = true }
      else if trimmed = "false" then
        Ok { title = strip title; author = strip author; year; isbn = strip isbn; available = false }
      else
        Error (Printf.sprintf "Invalid boolean: %s" avail_str)
