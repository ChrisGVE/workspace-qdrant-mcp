defmodule Bookshelf.Storage do
  @moduledoc "Storage operations for managing books on shelves."

  alias Bookshelf.Models.{Book, Shelf}

  @doc "Check whether the shelf is at capacity."
  def is_full(%Shelf{capacity: capacity, books: books}) do
    length(books) >= capacity
  end

  @doc "Add a book to the shelf."
  def add_book(%Shelf{} = shelf, %Book{} = book) do
    if is_full(shelf) do
      {:error, "Shelf '#{shelf.name}' is at capacity (#{shelf.capacity})"}
    else
      {:ok, %{shelf | books: shelf.books ++ [book]}}
    end
  end

  @doc "Remove a book by ISBN."
  def remove_book(%Shelf{} = shelf, isbn) do
    case Enum.split_while(shelf.books, fn b -> b.isbn != isbn end) do
      {_before, []} ->
        {:error, "No book with ISBN '#{isbn}' found on shelf"}
      {before, [found | after_books]} ->
        {:ok, %{shelf | books: before ++ after_books}, found}
    end
  end

  @doc "Find books by author (case-insensitive substring match)."
  def find_by_author(%Shelf{books: books}, query) do
    lq = String.downcase(query)
    Enum.filter(books, fn b ->
      String.contains?(String.downcase(b.author), lq)
    end)
  end

  @doc "Find books within an inclusive year range."
  def find_by_year_range(%Shelf{books: books}, start_year, end_year) do
    Enum.filter(books, fn b -> b.year >= start_year and b.year <= end_year end)
  end

  @doc "Return books sorted by title without mutating."
  def sort_by_title(%Shelf{books: books}) do
    Enum.sort_by(books, fn b -> b.title end)
  end

  @doc "Generate a formatted report for the shelf."
  def generate_report(%Shelf{} = shelf) do
    books = shelf.books
    total = length(books)
    avail_count = Enum.count(books, fn b -> b.available end)
    avail_pct = if total > 0, do: div(avail_count * 100, total), else: 0
    cap_pct = if shelf.capacity > 0, do: div(total * 100, shelf.capacity), else: 0

    author_counts = count_authors(books)
    unique_count = length(author_counts)

    years = Enum.map(books, fn b -> b.year end)
    min_year = Enum.min(years)
    max_year = Enum.max(years)

    author_lines = Enum.map(author_counts, fn {a, c} ->
      "  - #{a} (#{c} books)"
    end)

    book_lines = Enum.map(books, fn b ->
      marker = if b.available, do: "+", else: "-"
      "  [#{marker}] #{b.title} by #{b.author} (#{b.year}) - #{b.isbn}"
    end)

    lines = [
      "=== Library Report: #{shelf.name} ===",
      "Total books: #{total}",
      "Available: #{avail_count} / #{total} (#{avail_pct}%)",
      "Capacity: #{total} / #{shelf.capacity} (#{cap_pct}% full)",
      "",
      "Authors (#{unique_count} unique):"
    ] ++ author_lines ++ [
      "",
      "Year range: #{min_year} - #{max_year}",
      "",
      "Books by availability:"
    ] ++ book_lines

    Enum.join(lines, "\n")
  end

  defp count_authors(books) do
    Enum.reduce(books, {[], []}, fn b, {seen, acc} ->
      if b.author in seen do
        {seen, acc}
      else
        count = Enum.count(books, fn x -> x.author == b.author end)
        {[b.author | seen], acc ++ [{b.author, count}]}
      end
    end)
    |> elem(1)
  end
end
