%% @doc Storage operations for managing books on shelves.
-module(storage).
-export([add_book/2, remove_book/2, find_by_author/2, find_by_year_range/3,
         sort_by_title/1, is_full/1, generate_report/1]).

%% @doc Check whether the shelf is at capacity.
-spec is_full(models:shelf()) -> boolean().
is_full(#{capacity := Capacity, books := Books}) ->
    length(Books) >= Capacity.

%% @doc Add a book to the shelf.
-spec add_book(models:shelf(), models:book()) -> {ok, models:shelf()} | {error, string()}.
add_book(Shelf = #{name := Name, capacity := Capacity, books := Books}, Book) ->
    case is_full(Shelf) of
        true ->
            {error, lists:flatten(io_lib:format("Shelf '~s' is at capacity (~B)", [Name, Capacity]))};
        false ->
            {ok, Shelf#{books => Books ++ [Book]}}
    end.

%% @doc Remove a book by ISBN.
-spec remove_book(models:shelf(), string()) -> {ok, models:shelf(), models:book()} | {error, string()}.
remove_book(Shelf = #{books := Books}, Isbn) ->
    case lists:splitwith(fun(B) -> maps:get(isbn, B) =/= Isbn end, Books) of
        {_Before, []} ->
            {error, lists:flatten(io_lib:format("No book with ISBN '~s' found on shelf", [Isbn]))};
        {Before, [Found | After]} ->
            {ok, Shelf#{books => Before ++ After}, Found}
    end.

%% @doc Find books by author (case-insensitive substring match).
-spec find_by_author(models:shelf(), string()) -> [models:book()].
find_by_author(#{books := Books}, Query) ->
    LowerQuery = string:lowercase(Query),
    lists:filter(fun(B) ->
        LowerAuthor = string:lowercase(maps:get(author, B)),
        string:find(LowerAuthor, LowerQuery) =/= nomatch
    end, Books).

%% @doc Find books within an inclusive year range.
-spec find_by_year_range(models:shelf(), integer(), integer()) -> [models:book()].
find_by_year_range(#{books := Books}, StartYear, EndYear) ->
    lists:filter(fun(B) ->
        Y = maps:get(year, B),
        Y >= StartYear andalso Y =< EndYear
    end, Books).

%% @doc Return books sorted by title without mutating.
-spec sort_by_title(models:shelf()) -> [models:book()].
sort_by_title(#{books := Books}) ->
    lists:sort(fun(A, B) -> maps:get(title, A) =< maps:get(title, B) end, Books).

%% @doc Generate a formatted report for the shelf.
-spec generate_report(models:shelf()) -> string().
generate_report(#{name := Name, capacity := Capacity, books := Books}) ->
    Total = length(Books),
    AvailCount = length(lists:filter(fun(B) -> maps:get(available, B) end, Books)),
    AvailPct = case Total > 0 of true -> (AvailCount * 100) div Total; false -> 0 end,
    CapPct = case Capacity > 0 of true -> (Total * 100) div Capacity; false -> 0 end,
    AuthorCounts = count_authors(Books),
    UniqueCount = length(AuthorCounts),
    Years = [maps:get(year, B) || B <- Books],
    MinYear = lists:min(Years),
    MaxYear = lists:max(Years),
    Header = io_lib:format("=== Library Report: ~s ===", [Name]),
    AuthorLines = lists:map(fun({A, C}) ->
        io_lib:format("  - ~s (~B books)", [A, C])
    end, AuthorCounts),
    BookLines = lists:map(fun(B) ->
        Marker = case maps:get(available, B) of true -> "+"; false -> "-" end,
        io_lib:format("  [~s] ~s by ~s (~B) - ~s",
            [Marker, maps:get(title, B), maps:get(author, B),
             maps:get(year, B), maps:get(isbn, B)])
    end, Books),
    Lines = [Header,
             io_lib:format("Total books: ~B", [Total]),
             io_lib:format("Available: ~B / ~B (~B%)", [AvailCount, Total, AvailPct]),
             io_lib:format("Capacity: ~B / ~B (~B% full)", [Total, Capacity, CapPct]),
             "",
             io_lib:format("Authors (~B unique):", [UniqueCount])]
            ++ AuthorLines ++
            ["",
             io_lib:format("Year range: ~B - ~B", [MinYear, MaxYear]),
             "",
             "Books by availability:"]
            ++ BookLines,
    lists:flatten(lists:join("\n", Lines)).

%% @doc Count authors preserving insertion order.
count_authors(Books) ->
    count_authors(Books, Books, [], []).

count_authors([], _All, _Seen, Acc) ->
    lists:reverse(Acc);
count_authors([B | Rest], All, Seen, Acc) ->
    Author = maps:get(author, B),
    case lists:member(Author, Seen) of
        true -> count_authors(Rest, All, Seen, Acc);
        false ->
            AuthorCount = length(lists:filter(fun(X) -> maps:get(author, X) =:= Author end, All)),
            count_authors(Rest, All, [Author | Seen], [{Author, AuthorCount} | Acc])
    end.
