module Genre
  FICTION = "Fiction"
  NON_FICTION = "NonFiction"
  SCIENCE = "Science"
  HISTORY = "History"
  BIOGRAPHY = "Biography"
  TECHNOLOGY = "Technology"
  PHILOSOPHY = "Philosophy"
end

class Book
  attr_reader :title, :author, :year, :isbn, :available

  def initialize(title, author, year, isbn, available)
    @title = title
    @author = author
    @year = year
    @isbn = isbn
    @available = available
  end
end

class Shelf
  attr_reader :name, :capacity
  attr_accessor :books

  def initialize(name, capacity)
    @name = name
    @capacity = capacity
    @books = []
  end
end
