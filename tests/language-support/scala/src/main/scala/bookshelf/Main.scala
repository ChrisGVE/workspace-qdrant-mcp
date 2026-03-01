package bookshelf

/** Bookshelf demonstration: creates a shelf, adds books, and runs queries. */
@main def main(): Unit =
  var shelf = Shelf("Computer Science", 10)

  val books = List(
    Book("The Art of Computer Programming", "Donald Knuth", 1968, "9780201896831", true),
    Book("Structure and Interpretation of Computer Programs", "Harold Abelson", 1996, "9780262510875", true),
    Book("Introduction to Algorithms", "Thomas Cormen", 2009, "9780262033848", false),
    Book("Design Patterns", "Erich Gamma", 1994, "9780201633610", true),
    Book("The Pragmatic Programmer", "David Thomas", 2019, "9780135957059", true)
  )

  for book <- books do
    Storage.addBook(shelf, book) match
      case Right(s) => shelf = s
      case Left(_) => ()

  println(Storage.generateReport(shelf))
  println()

  println("--- Search by author \"knuth\" ---")
  for book <- Storage.findByAuthor(shelf, "knuth") do
    println(s"  ${Utils.formatBook(book)}")
  println()

  println("--- Search by year range 1990-2010 ---")
  for book <- Storage.findByYearRange(shelf, 1990, 2010) do
    println(s"  ${Utils.formatBook(book)}")
  println()

  println("--- Parse CSV ---")
  Utils.parseCsvLine("Clean Code,Robert Martin,2008,9780132350884,true") match
    case Right(parsed) => println(s"  Parsed: ${Utils.formatBook(parsed)}")
    case Left(err) => println(s"  Error: $err")
  println()

  println("--- ISBN Validation ---")
  for book <- books do
    val status = if Utils.validateIsbn(book.isbn) then "valid" else "invalid"
    println(s"  ${book.isbn}: $status")
