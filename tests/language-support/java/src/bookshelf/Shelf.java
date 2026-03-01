package bookshelf;

import java.util.ArrayList;
import java.util.List;

public class Shelf {
    public final String name;
    public final int capacity;
    public final List<Book> books;

    public Shelf(String name, int capacity) {
        this.name = name;
        this.capacity = capacity;
        this.books = new ArrayList<>();
    }
}
