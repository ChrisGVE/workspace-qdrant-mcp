package bookshelf

/** Storage operations for managing books on shelves. */
object Storage:

  /** Check whether the shelf is at capacity. */
  def isFull(shelf: Shelf): Boolean =
    shelf.books.length >= shelf.capacity

  /** Add a book to the shelf. */
  def addBook(shelf: Shelf, book: Book): Either[String, Shelf] =
    if isFull(shelf) then
      Left(s"Shelf '${shelf.name}' is at capacity (${shelf.capacity})")
    else
      Right(shelf.copy(books = shelf.books :+ book))

  /** Remove a book by ISBN. */
  def removeBook(shelf: Shelf, isbn: String): Either[String, (Shelf, Book)] =
    shelf.books.indexWhere(_.isbn == isbn) match
      case -1 => Left(s"No book with ISBN '$isbn' found on shelf")
      case i =>
        val book = shelf.books(i)
        val remaining = shelf.books.take(i) ++ shelf.books.drop(i + 1)
        Right((shelf.copy(books = remaining), book))

  /** Find books by author (case-insensitive substring match). */
  def findByAuthor(shelf: Shelf, query: String): List[Book] =
    val lq = query.toLowerCase
    shelf.books.filter(b => b.author.toLowerCase.contains(lq))

  /** Find books within an inclusive year range. */
  def findByYearRange(shelf: Shelf, startYear: Int, endYear: Int): List[Book] =
    shelf.books.filter(b => b.year >= startYear && b.year <= endYear)

  /** Return books sorted by title without mutating. */
  def sortByTitle(shelf: Shelf): List[Book] =
    shelf.books.sortBy(_.title)

  /** Generate a formatted report for the shelf. */
  def generateReport(shelf: Shelf): String =
    val books = shelf.books
    val total = books.length
    val availCount = books.count(_.available)
    val availPct = if total > 0 then (availCount * 100) / total else 0
    val capPct = if shelf.capacity > 0 then (total * 100) / shelf.capacity else 0

    val authorCounts = countAuthors(books)
    val uniqueCount = authorCounts.length

    val years = books.map(_.year)
    val minYear = years.min
    val maxYear = years.max

    val authorLines = authorCounts.map((a, c) => s"  - $a ($c books)")

    val bookLines = books.map { b =>
      val marker = if b.available then "+" else "-"
      s"  [$marker] ${b.title} by ${b.author} (${b.year}) - ${b.isbn}"
    }

    val lines = List(
      s"=== Library Report: ${shelf.name} ===",
      s"Total books: $total",
      s"Available: $availCount / $total ($availPct%)",
      s"Capacity: $total / ${shelf.capacity} ($capPct% full)",
      "",
      s"Authors ($uniqueCount unique):"
    ) ++ authorLines ++ List(
      "",
      s"Year range: $minYear - $maxYear",
      "",
      "Books by availability:"
    ) ++ bookLines

    lines.mkString("\n")

  private def countAuthors(books: List[Book]): List[(String, Int)] =
    books.foldLeft((Set.empty[String], List.empty[(String, Int)])) { case ((seen, acc), b) =>
      if seen.contains(b.author) then (seen, acc)
      else
        val count = books.count(_.author == b.author)
        (seen + b.author, acc :+ (b.author, count))
    }._2
