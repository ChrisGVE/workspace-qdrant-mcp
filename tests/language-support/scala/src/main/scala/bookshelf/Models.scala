package bookshelf

/** Genre classification for books. */
enum Genre:
  case Fiction, NonFiction, Science, History, Biography, Technology, Philosophy

/** Represents a single book in the library. */
case class Book(
  title: String,
  author: String,
  year: Int,
  isbn: String,
  available: Boolean
)

/** Represents a bookshelf with a fixed capacity. */
case class Shelf(
  name: String,
  capacity: Int,
  books: List[Book] = List.empty
)
