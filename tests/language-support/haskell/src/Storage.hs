-- | Storage operations for managing books on shelves.
module Storage
    ( addBook
    , removeBook
    , findByAuthor
    , findByYearRange
    , sortByTitle
    , isFull
    , generateReport
    ) where

import Data.Char (toLower)
import Data.List (sortBy, isInfixOf, intercalate, nub)
import Models

-- | Add a book to the shelf. Returns Left on error.
addBook :: Shelf -> Book -> Either String Shelf
addBook shelf book
    | isFull shelf = Left $ "Shelf '" ++ shelfName shelf ++ "' is at capacity (" ++ show (shelfCapacity shelf) ++ ")"
    | otherwise    = Right shelf { shelfBooks = shelfBooks shelf ++ [book] }

-- | Remove a book by ISBN. Returns Left on error.
removeBook :: Shelf -> String -> Either String (Shelf, Book)
removeBook shelf isbn =
    case break (\b -> bookIsbn b == isbn) (shelfBooks shelf) of
        (_, [])    -> Left $ "No book with ISBN '" ++ isbn ++ "' found on shelf"
        (before, b:after) -> Right (shelf { shelfBooks = before ++ after }, b)

-- | Find books by author (case-insensitive substring match).
findByAuthor :: Shelf -> String -> [Book]
findByAuthor shelf query =
    filter (\b -> map toLower query `isInfixOf` map toLower (bookAuthor b)) (shelfBooks shelf)

-- | Find books within an inclusive year range.
findByYearRange :: Shelf -> Int -> Int -> [Book]
findByYearRange shelf startYear endYear =
    filter (\b -> bookYear b >= startYear && bookYear b <= endYear) (shelfBooks shelf)

-- | Return books sorted by title without mutating.
sortByTitle :: Shelf -> [Book]
sortByTitle shelf = sortBy (\a b -> compare (bookTitle a) (bookTitle b)) (shelfBooks shelf)

-- | Check whether the shelf is at capacity.
isFull :: Shelf -> Bool
isFull shelf = length (shelfBooks shelf) >= shelfCapacity shelf

-- | Generate a formatted report for the shelf.
generateReport :: Shelf -> String
generateReport shelf =
    let books = shelfBooks shelf
        total = length books
        availCount = length (filter bookAvailable books)
        availPct = if total > 0 then (availCount * 100) `div` total else 0
        capPct = if shelfCapacity shelf > 0 then (total * 100) `div` shelfCapacity shelf else 0
        authors = map bookAuthor books
        uniqueAuthors = nub authors
        uniqueCount = length uniqueAuthors
        authorCounts = map (\a -> (a, length (filter (== a) authors))) uniqueAuthors
        years = map bookYear books
        minYear = minimum years
        maxYear = maximum years
        header = "=== Library Report: " ++ shelfName shelf ++ " ==="
        bookLines = map formatAvailLine books
        authorLines = map (\(a, c) -> "  - " ++ a ++ " (" ++ show c ++ " books)") authorCounts
    in intercalate "\n" $
        [ header
        , "Total books: " ++ show total
        , "Available: " ++ show availCount ++ " / " ++ show total ++ " (" ++ show availPct ++ "%)"
        , "Capacity: " ++ show total ++ " / " ++ show (shelfCapacity shelf) ++ " (" ++ show capPct ++ "% full)"
        , ""
        , "Authors (" ++ show uniqueCount ++ " unique):"
        ] ++ authorLines ++
        [ ""
        , "Year range: " ++ show minYear ++ " - " ++ show maxYear
        , ""
        , "Books by availability:"
        ] ++ bookLines

formatAvailLine :: Book -> String
formatAvailLine book =
    let marker = if bookAvailable book then "+" else "-"
    in "  [" ++ marker ++ "] " ++ bookTitle book ++ " by " ++ bookAuthor book
       ++ " (" ++ show (bookYear book) ++ ") - " ++ bookIsbn book
