-- | Data models for the bookshelf library management system.
module Models
    ( Book(..)
    , Shelf(..)
    , Genre(..)
    , newShelf
    ) where

-- | Genre classification for books.
data Genre
    = Fiction
    | NonFiction
    | Science
    | History
    | Biography
    | Technology
    | Philosophy
    deriving (Show, Eq)

-- | Represents a single book in the library.
data Book = Book
    { bookTitle     :: String
    , bookAuthor    :: String
    , bookYear      :: Int
    , bookIsbn      :: String
    , bookAvailable :: Bool
    } deriving (Show, Eq)

-- | Represents a bookshelf with a fixed capacity.
data Shelf = Shelf
    { shelfName     :: String
    , shelfCapacity :: Int
    , shelfBooks    :: [Book]
    } deriving (Show, Eq)

-- | Create a new empty shelf.
newShelf :: String -> Int -> Shelf
newShelf name capacity = Shelf name capacity []
