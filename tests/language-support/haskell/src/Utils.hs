-- | Utility functions for ISBN validation, formatting, and CSV parsing.
module Utils
    ( validateIsbn
    , formatBook
    , parseCsvLine
    ) where

import Data.Char (isDigit, digitToInt)
import Models

-- | Validate an ISBN-13 check digit.
validateIsbn :: String -> Bool
validateIsbn isbn
    | length isbn /= 13 = False
    | not (all isDigit isbn) = False
    | otherwise =
        let digits = map digitToInt isbn
            total = sum $ zipWith (*) digits (cycle [1, 3])
        in total `mod` 10 == 0

-- | Format a book as a single descriptive line.
formatBook :: Book -> String
formatBook book =
    "\"" ++ bookTitle book ++ "\" by " ++ bookAuthor book
    ++ " (" ++ show (bookYear book) ++ ") [ISBN: " ++ bookIsbn book ++ "]"

-- | Parse a CSV line into a Book.
parseCsvLine :: String -> Either String Book
parseCsvLine line =
    let parts = splitOn ',' line
    in if length parts /= 5
       then Left $ "Expected 5 fields, got " ++ show (length parts)
       else let [title, author, yearStr, isbn, availStr] = parts
                trimmed = strip availStr
            in case reads yearStr :: [(Int, String)] of
                 [(year, "")] ->
                     case toLowerStr trimmed of
                         "true"  -> Right $ Book (strip title) (strip author) year (strip isbn) True
                         "false" -> Right $ Book (strip title) (strip author) year (strip isbn) False
                         _       -> Left $ "Invalid boolean: " ++ availStr
                 _ -> Left $ "Invalid year: " ++ yearStr

-- | Split a string on a delimiter.
splitOn :: Char -> String -> [String]
splitOn _ [] = [""]
splitOn delim s =
    let (first, rest) = break (== delim) s
    in first : case rest of
        []     -> []
        (_:rs) -> splitOn delim rs

-- | Strip leading and trailing whitespace.
strip :: String -> String
strip = reverse . dropWhile (== ' ') . reverse . dropWhile (== ' ')

-- | Convert string to lowercase.
toLowerStr :: String -> String
toLowerStr = map toLowerChar
  where
    toLowerChar c
        | c >= 'A' && c <= 'Z' = toEnum (fromEnum c + 32)
        | otherwise = c
