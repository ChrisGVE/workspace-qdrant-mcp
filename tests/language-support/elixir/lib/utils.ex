defmodule Bookshelf.Utils do
  @moduledoc "Utility functions for ISBN validation, formatting, and CSV parsing."

  alias Bookshelf.Models.Book

  @doc "Validate an ISBN-13 check digit."
  def validate_isbn(isbn) do
    digits = String.graphemes(isbn)
    cond do
      String.length(isbn) != 13 -> false
      not Enum.all?(digits, fn c -> c >= "0" and c <= "9" end) -> false
      true ->
        total = digits
        |> Enum.with_index()
        |> Enum.reduce(0, fn {c, i}, acc ->
          d = String.to_integer(c)
          w = if rem(i, 2) == 0, do: 1, else: 3
          acc + d * w
        end)
        rem(total, 10) == 0
    end
  end

  @doc "Format a book as a single descriptive line."
  def format_book(%Book{} = book) do
    "\"#{book.title}\" by #{book.author} (#{book.year}) [ISBN: #{book.isbn}]"
  end

  @doc "Parse a CSV line into a Book."
  def parse_csv_line(line) do
    parts = String.split(line, ",")
    case length(parts) do
      5 ->
        [title, author, year_str, isbn, avail_str] = parts
        case Integer.parse(String.trim(year_str)) do
          {year, ""} ->
            trimmed = String.downcase(String.trim(avail_str))
            case trimmed do
              "true" ->
                {:ok, %Book{title: String.trim(title), author: String.trim(author),
                            year: year, isbn: String.trim(isbn), available: true}}
              "false" ->
                {:ok, %Book{title: String.trim(title), author: String.trim(author),
                            year: year, isbn: String.trim(isbn), available: false}}
              _ ->
                {:error, "Invalid boolean: #{avail_str}"}
            end
          _ ->
            {:error, "Invalid year: #{year_str}"}
        end
      n ->
        {:error, "Expected 5 fields, got #{n}"}
    end
  end
end
