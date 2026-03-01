require_relative "lib/models"
require_relative "lib/storage"
require_relative "lib/utils"

shelf = Shelf.new("Computer Science", 10)

books = [
  Book.new("The Art of Computer Programming", "Donald Knuth",
           1968, "9780201896831", true),
  Book.new("Structure and Interpretation of Computer Programs",
           "Harold Abelson", 1996, "9780262510875", true),
  Book.new("Introduction to Algorithms", "Thomas Cormen",
           2009, "9780262033848", false),
  Book.new("Design Patterns", "Erich Gamma",
           1994, "9780201633610", true),
  Book.new("The Pragmatic Programmer", "David Thomas",
           2019, "9780135957059", true),
]

books.each { |book| add_book(shelf, book) }

puts generate_report(shelf)

puts
puts '--- Search by author "knuth" ---'
find_by_author(shelf, "knuth").each do |book|
  puts "  #{format_book(book)}"
end
puts

puts "--- Search by year range 1990-2010 ---"
find_by_year_range(shelf, 1990, 2010).each do |book|
  puts "  #{format_book(book)}"
end
puts

puts "--- Parse CSV ---"
parsed = parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true")
puts "  Parsed: #{format_book(parsed)}"
puts

puts "--- ISBN Validation ---"
books.each do |book|
  status = validate_isbn(book.isbn) ? "valid" : "invalid"
  puts "  #{book.isbn}: #{status}"
end
