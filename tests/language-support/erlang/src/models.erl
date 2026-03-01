%% @doc Data models for the bookshelf library management system.
-module(models).
-export([new_book/5, new_shelf/2, genre_values/0]).

%% Genre classification for books.
-type genre() :: fiction | non_fiction | science | history | biography | technology | philosophy.

%% Book record stored as a map.
-type book() :: #{title := string(), author := string(), year := integer(),
                  isbn := string(), available := boolean()}.

%% Shelf record stored as a map.
-type shelf() :: #{name := string(), capacity := integer(), books := [book()]}.

-export_type([genre/0, book/0, shelf/0]).

%% @doc Create a new book.
-spec new_book(string(), string(), integer(), string(), boolean()) -> book().
new_book(Title, Author, Year, Isbn, Available) ->
    #{title => Title, author => Author, year => Year,
      isbn => Isbn, available => Available}.

%% @doc Create a new empty shelf.
-spec new_shelf(string(), integer()) -> shelf().
new_shelf(Name, Capacity) ->
    #{name => Name, capacity => Capacity, books => []}.

%% @doc Return all genre values.
-spec genre_values() -> [genre()].
genre_values() ->
    [fiction, non_fiction, science, history, biography, technology, philosophy].
