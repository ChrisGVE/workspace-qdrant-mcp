%% @doc Bookshelf demonstration: creates a shelf, adds books, and runs queries.
-module(main).
-export([main/0]).

main() ->
    Shelf0 = models:new_shelf("Computer Science", 10),
    Books = [
        models:new_book("The Art of Computer Programming", "Donald Knuth", 1968, "9780201896831", true),
        models:new_book("Structure and Interpretation of Computer Programs", "Harold Abelson", 1996, "9780262510875", true),
        models:new_book("Introduction to Algorithms", "Thomas Cormen", 2009, "9780262033848", false),
        models:new_book("Design Patterns", "Erich Gamma", 1994, "9780201633610", true),
        models:new_book("The Pragmatic Programmer", "David Thomas", 2019, "9780135957059", true)
    ],
    Shelf = lists:foldl(fun(Book, S) ->
        case storage:add_book(S, Book) of
            {ok, S2} -> S2;
            {error, _} -> S
        end
    end, Shelf0, Books),

    io:format("~s~n", [storage:generate_report(Shelf)]),
    io:format("~n", []),

    io:format("--- Search by author \"knuth\" ---~n", []),
    lists:foreach(fun(B) ->
        io:format("  ~s~n", [utils:format_book(B)])
    end, storage:find_by_author(Shelf, "knuth")),
    io:format("~n", []),

    io:format("--- Search by year range 1990-2010 ---~n", []),
    lists:foreach(fun(B) ->
        io:format("  ~s~n", [utils:format_book(B)])
    end, storage:find_by_year_range(Shelf, 1990, 2010)),
    io:format("~n", []),

    io:format("--- Parse CSV ---~n", []),
    case utils:parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true") of
        {ok, Parsed} ->
            io:format("  Parsed: ~s~n", [utils:format_book(Parsed)]);
        {error, Err} ->
            io:format("  Error: ~s~n", [Err])
    end,
    io:format("~n", []),

    io:format("--- ISBN Validation ---~n", []),
    lists:foreach(fun(B) ->
        Status = case utils:validate_isbn(maps:get(isbn, B)) of
            true -> "valid";
            false -> "invalid"
        end,
        io:format("  ~s: ~s~n", [maps:get(isbn, B), Status])
    end, Books).
