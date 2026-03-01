%% @doc Utility functions for ISBN validation, formatting, and CSV parsing.
-module(utils).
-export([validate_isbn/1, format_book/1, parse_csv_line/1]).

%% @doc Validate an ISBN-13 check digit.
-spec validate_isbn(string()) -> boolean().
validate_isbn(Isbn) ->
    case length(Isbn) =:= 13 andalso lists:all(fun(C) -> C >= $0 andalso C =< $9 end, Isbn) of
        false -> false;
        true ->
            Total = lists:foldl(fun({I, C}, Acc) ->
                D = C - $0,
                W = case I rem 2 of 0 -> 1; _ -> 3 end,
                Acc + D * W
            end, 0, lists:zip(lists:seq(0, 12), Isbn)),
            Total rem 10 =:= 0
    end.

%% @doc Format a book as a single descriptive line.
-spec format_book(models:book()) -> string().
format_book(Book) ->
    lists:flatten(io_lib:format("\"~s\" by ~s (~B) [ISBN: ~s]",
        [maps:get(title, Book), maps:get(author, Book),
         maps:get(year, Book), maps:get(isbn, Book)])).

%% @doc Parse a CSV line into a Book.
-spec parse_csv_line(string()) -> {ok, models:book()} | {error, string()}.
parse_csv_line(Line) ->
    Parts = string:split(Line, ",", all),
    case length(Parts) of
        5 ->
            [Title, Author, YearStr, Isbn, AvailStr] = Parts,
            case catch list_to_integer(string:trim(YearStr)) of
                Year when is_integer(Year) ->
                    Trimmed = string:lowercase(string:trim(AvailStr)),
                    case Trimmed of
                        "true" ->
                            {ok, models:new_book(string:trim(Title), string:trim(Author),
                                                 Year, string:trim(Isbn), true)};
                        "false" ->
                            {ok, models:new_book(string:trim(Title), string:trim(Author),
                                                 Year, string:trim(Isbn), false)};
                        _ ->
                            {error, lists:flatten(io_lib:format("Invalid boolean: ~s", [AvailStr]))}
                    end;
                _ ->
                    {error, lists:flatten(io_lib:format("Invalid year: ~s", [YearStr]))}
            end;
        N ->
            {error, lists:flatten(io_lib:format("Expected 5 fields, got ~B", [N]))}
    end.
