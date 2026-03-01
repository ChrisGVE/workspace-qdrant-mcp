package bookshelf

/** Utility functions for ISBN validation, formatting, and CSV parsing. */
object Utils:

  /** Validate an ISBN-13 check digit. */
  def validateIsbn(isbn: String): Boolean =
    if isbn.length != 13 || !isbn.forall(_.isDigit) then false
    else
      val total = isbn.zipWithIndex.map { (c, i) =>
        val d = c.asDigit
        val w = if i % 2 == 0 then 1 else 3
        d * w
      }.sum
      total % 10 == 0

  /** Format a book as a single descriptive line. */
  def formatBook(book: Book): String =
    s""""${book.title}" by ${book.author} (${book.year}) [ISBN: ${book.isbn}]"""

  /** Parse a CSV line into a Book. */
  def parseCsvLine(line: String): Either[String, Book] =
    val parts = line.split(",", -1).toList
    if parts.length != 5 then
      Left(s"Expected 5 fields, got ${parts.length}")
    else
      val List(title, author, yearStr, isbn, availStr) = parts: @unchecked
      yearStr.trim.toIntOption match
        case None => Left(s"Invalid year: $yearStr")
        case Some(year) =>
          availStr.trim.toLowerCase match
            case "true" =>
              Right(Book(title.trim, author.trim, year, isbn.trim, true))
            case "false" =>
              Right(Book(title.trim, author.trim, year, isbn.trim, false))
            case _ =>
              Left(s"Invalid boolean: $availStr")
