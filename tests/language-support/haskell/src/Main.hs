-- | Bookshelf demonstration: creates a shelf, adds books, and runs queries.
module Main (main) where

import Models
import Storage
import Utils

main :: IO ()
main = do
    let shelf0 = newShelf "Computer Science" 10
        books =
            [ Book "The Art of Computer Programming" "Donald Knuth" 1968 "9780201896831" True
            , Book "Structure and Interpretation of Computer Programs" "Harold Abelson" 1996 "9780262510875" True
            , Book "Introduction to Algorithms" "Thomas Cormen" 2009 "9780262033848" False
            , Book "Design Patterns" "Erich Gamma" 1994 "9780201633610" True
            , Book "The Pragmatic Programmer" "David Thomas" 2019 "9780135957059" True
            ]
        shelf = foldl (\s b -> case addBook s b of Right s' -> s'; Left _ -> s) shelf0 books

    putStrLn (generateReport shelf)
    putStrLn ""

    putStrLn "--- Search by author \"knuth\" ---"
    mapM_ (\b -> putStrLn $ "  " ++ formatBook b) (findByAuthor shelf "knuth")
    putStrLn ""

    putStrLn "--- Search by year range 1990-2010 ---"
    mapM_ (\b -> putStrLn $ "  " ++ formatBook b) (findByYearRange shelf 1990 2010)
    putStrLn ""

    putStrLn "--- Parse CSV ---"
    case parseCsvLine "Clean Code,Robert Martin,2008,9780132350884,true" of
        Right book -> putStrLn $ "  Parsed: " ++ formatBook book
        Left err   -> putStrLn $ "  Error: " ++ err
    putStrLn ""

    putStrLn "--- ISBN Validation ---"
    mapM_ (\b -> do
        let status = if validateIsbn (bookIsbn b) then "valid" else "invalid"
        putStrLn $ "  " ++ bookIsbn b ++ ": " ++ status
        ) books
