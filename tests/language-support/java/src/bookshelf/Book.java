package bookshelf;

public class Book {
    public final String title;
    public final String author;
    public final int year;
    public final String isbn;
    public final boolean available;

    public Book(String title, String author, int year,
                String isbn, boolean available) {
        this.title = title;
        this.author = author;
        this.year = year;
        this.isbn = isbn;
        this.available = available;
    }
}
