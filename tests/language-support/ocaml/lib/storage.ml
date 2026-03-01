(** Storage operations for managing books on shelves. *)

open Models

(** Check whether the shelf is at capacity. *)
let is_full shelf = List.length shelf.books >= shelf.capacity

(** Add a book to the shelf. *)
let add_book shelf book =
  if is_full shelf then
    Error (Printf.sprintf "Shelf '%s' is at capacity (%d)" shelf.name shelf.capacity)
  else begin
    shelf.books <- shelf.books @ [book];
    Ok ()
  end

(** Remove a book by ISBN. *)
let remove_book shelf isbn =
  let rec aux acc = function
    | [] -> Error (Printf.sprintf "No book with ISBN '%s' found on shelf" isbn)
    | b :: rest when b.isbn = isbn ->
      shelf.books <- List.rev acc @ rest;
      Ok b
    | b :: rest -> aux (b :: acc) rest
  in
  aux [] shelf.books

(** Find books by author (case-insensitive substring match). *)
let find_by_author shelf query =
  let lq = String.lowercase_ascii query in
  List.filter (fun b ->
    let la = String.lowercase_ascii b.author in
    let qlen = String.length lq in
    let alen = String.length la in
    if qlen > alen then false
    else
      let found = ref false in
      for i = 0 to alen - qlen do
        if String.sub la i qlen = lq then found := true
      done;
      !found
  ) shelf.books

(** Find books within an inclusive year range. *)
let find_by_year_range shelf start_year end_year =
  List.filter (fun b -> b.year >= start_year && b.year <= end_year) shelf.books

(** Return books sorted by title without mutating. *)
let sort_by_title shelf =
  List.sort (fun a b -> String.compare a.title b.title) shelf.books

(** Count occurrences preserving insertion order of unique elements. *)
let count_authors books =
  let seen = ref [] in
  List.iter (fun b ->
    if not (List.mem b.author (List.map fst !seen)) then
      seen := !seen @ [(b.author, List.length (List.filter (fun x -> x.author = b.author) books))]
  ) books;
  !seen

(** Generate a formatted report for the shelf. *)
let generate_report shelf =
  let books = shelf.books in
  let total = List.length books in
  let avail_count = List.length (List.filter (fun b -> b.available) books) in
  let avail_pct = if total > 0 then (avail_count * 100) / total else 0 in
  let cap_pct = if shelf.capacity > 0 then (total * 100) / shelf.capacity else 0 in
  let author_counts = count_authors books in
  let unique_count = List.length author_counts in
  let years = List.map (fun b -> b.year) books in
  let min_year = List.fold_left min max_int years in
  let max_year = List.fold_left max min_int years in
  let buf = Buffer.create 512 in
  let add s = Buffer.add_string buf s in
  let addln s = add s; add "\n" in
  addln (Printf.sprintf "=== Library Report: %s ===" shelf.name);
  addln (Printf.sprintf "Total books: %d" total);
  addln (Printf.sprintf "Available: %d / %d (%d%%)" avail_count total avail_pct);
  addln (Printf.sprintf "Capacity: %d / %d (%d%% full)" total shelf.capacity cap_pct);
  addln "";
  addln (Printf.sprintf "Authors (%d unique):" unique_count);
  List.iter (fun (a, c) ->
    addln (Printf.sprintf "  - %s (%d books)" a c)
  ) author_counts;
  addln "";
  addln (Printf.sprintf "Year range: %d - %d" min_year max_year);
  addln "";
  addln "Books by availability:";
  List.iter (fun b ->
    let marker = if b.available then "+" else "-" in
    addln (Printf.sprintf "  [%s] %s by %s (%d) - %s" marker b.title b.author b.year b.isbn)
  ) books;
  (* Remove trailing newline *)
  let s = Buffer.contents buf in
  if String.length s > 0 && s.[String.length s - 1] = '\n' then
    String.sub s 0 (String.length s - 1)
  else
    s
