defmodule Bookshelf do
  @moduledoc "Bookshelf demonstration: creates a shelf, adds books, and runs queries."

  alias Bookshelf.Models.{Book, Shelf}
  alias Bookshelf.Storage
  alias Bookshelf.Utils

  def main do
    shelf0 = %Shelf{name: "Computer Science", capacity: 10}

    books = [
      %Book{title: "The Art of Computer Programming", author: "Donald Knuth", year: 1968, isbn: "9780201896831", available: true},
      %Book{title: "Structure and Interpretation of Computer Programs", author: "Harold Abelson", year: 1996, isbn: "9780262510875", available: true},
      %Book{title: "Introduction to Algorithms", author: "Thomas Cormen", year: 2009, isbn: "9780262033848", available: false},
      %Book{title: "Design Patterns", author: "Erich Gamma", year: 1994, isbn: "9780201633610", available: true},
      %Book{title: "The Pragmatic Programmer", author: "David Thomas", year: 2019, isbn: "9780135957059", available: true}
    ]

    shelf = Enum.reduce(books, shelf0, fn book, s ->
      case Storage.add_book(s, book) do
        {:ok, s2} -> s2
        {:error, _} -> s
      end
    end)

    IO.puts(Storage.generate_report(shelf))
    IO.puts("")

    IO.puts("--- Search by author \"knuth\" ---")
    Enum.each(Storage.find_by_author(shelf, "knuth"), fn b ->
      IO.puts("  #{Utils.format_book(b)}")
    end)
    IO.puts("")

    IO.puts("--- Search by year range 1990-2010 ---")
    Enum.each(Storage.find_by_year_range(shelf, 1990, 2010), fn b ->
      IO.puts("  #{Utils.format_book(b)}")
    end)
    IO.puts("")

    IO.puts("--- Parse CSV ---")
    case Utils.parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true") do
      {:ok, parsed} -> IO.puts("  Parsed: #{Utils.format_book(parsed)}")
      {:error, err} -> IO.puts("  Error: #{err}")
    end
    IO.puts("")

    IO.puts("--- ISBN Validation ---")
    Enum.each(books, fn b ->
      status = if Utils.validate_isbn(b.isbn), do: "valid", else: "invalid"
      IO.puts("  #{b.isbn}: #{status}")
    end)
  end
end
