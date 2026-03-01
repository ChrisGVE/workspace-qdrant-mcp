require_relative "models"

def validate_isbn(isbn)
  return false unless isbn.length == 13
  return false unless isbn.match?(/\A\d{13}\z/)

  total = 0
  isbn.each_char.with_index do |ch, i|
    digit = ch.to_i
    total += i.even? ? digit : digit * 3
  end
  (total % 10).zero?
end

def format_book(book)
  "\"#{book.title}\" by #{book.author} (#{book.year}) [ISBN: #{book.isbn}]"
end

def parse_csv_line(line)
  parts = line.split(",", -1)
  raise "Expected 5 fields, got #{parts.length}" unless parts.length == 5

  title = parts[0].strip
  author = parts[1].strip
  year_str = parts[2].strip
  isbn = parts[3].strip
  avail_str = parts[4].strip.downcase

  year = Integer(year_str)

  case avail_str
  when "true"
    available = true
  when "false"
    available = false
  else
    raise "Invalid boolean: #{parts[4]}"
  end

  Book.new(title, author, year, isbn, available)
end
