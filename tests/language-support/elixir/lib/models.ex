defmodule Bookshelf.Models do
  @moduledoc "Data models for the bookshelf library management system."

  @type genre :: :fiction | :non_fiction | :science | :history | :biography | :technology | :philosophy

  @doc "All genre values."
  def genre_values do
    [:fiction, :non_fiction, :science, :history, :biography, :technology, :philosophy]
  end

  defmodule Book do
    @moduledoc "Represents a single book in the library."
    defstruct [:title, :author, :year, :isbn, :available]

    @type t :: %__MODULE__{
      title: String.t(),
      author: String.t(),
      year: integer(),
      isbn: String.t(),
      available: boolean()
    }
  end

  defmodule Shelf do
    @moduledoc "Represents a bookshelf with a fixed capacity."
    defstruct [:name, :capacity, books: []]

    @type t :: %__MODULE__{
      name: String.t(),
      capacity: integer(),
      books: [Bookshelf.Models.Book.t()]
    }
  end
end
