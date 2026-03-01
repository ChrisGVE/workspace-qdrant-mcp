package bookshelf;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Storage {

    public static void addBook(Shelf shelf, Book book) {
        if (isFull(shelf)) {
            throw new RuntimeException(
                "Shelf '" + shelf.name + "' is at capacity");
        }
        shelf.books.add(book);
    }

    public static Book removeBook(Shelf shelf, String isbn) {
        for (int i = 0; i < shelf.books.size(); i++) {
            if (shelf.books.get(i).isbn.equals(isbn)) {
                return shelf.books.remove(i);
            }
        }
        throw new RuntimeException(
            "No book with ISBN '" + isbn + "' found");
    }

    public static List<Book> findByAuthor(Shelf shelf, String author) {
        List<Book> results = new ArrayList<>();
        String query = author.toLowerCase();
        for (Book book : shelf.books) {
            if (book.author.toLowerCase().contains(query)) {
                results.add(book);
            }
        }
        return results;
    }

    public static List<Book> findByYearRange(Shelf shelf,
                                             int startYear, int endYear) {
        List<Book> results = new ArrayList<>();
        for (Book book : shelf.books) {
            if (book.year >= startYear && book.year <= endYear) {
                results.add(book);
            }
        }
        return results;
    }

    public static List<Book> sortByTitle(Shelf shelf) {
        List<Book> sorted = new ArrayList<>(shelf.books);
        sorted.sort((a, b) -> a.title.compareTo(b.title));
        return sorted;
    }

    public static boolean isFull(Shelf shelf) {
        return shelf.books.size() >= shelf.capacity;
    }

    public static String generateReport(Shelf shelf) {
        int total = shelf.books.size();
        int availableCount = 0;
        for (Book b : shelf.books) {
            if (b.available) availableCount++;
        }
        int availPct = total > 0 ? (availableCount * 100 / total) : 0;
        int capPct = shelf.capacity > 0
            ? (total * 100 / shelf.capacity) : 0;

        Map<String, Integer> authorCounts = new LinkedHashMap<>();
        for (Book b : shelf.books) {
            authorCounts.merge(b.author, 1, Integer::sum);
        }

        int minYear = shelf.books.get(0).year;
        int maxYear = shelf.books.get(0).year;
        for (Book b : shelf.books) {
            if (b.year < minYear) minYear = b.year;
            if (b.year > maxYear) maxYear = b.year;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("=== Library Report: ").append(shelf.name)
          .append(" ===\n");
        sb.append("Total books: ").append(total).append("\n");
        sb.append("Available: ").append(availableCount).append(" / ")
          .append(total).append(" (").append(availPct).append("%)\n");
        sb.append("Capacity: ").append(total).append(" / ")
          .append(shelf.capacity).append(" (").append(capPct)
          .append("% full)\n");
        sb.append("\n");
        sb.append("Authors (").append(authorCounts.size())
          .append(" unique):\n");
        for (Map.Entry<String, Integer> entry : authorCounts.entrySet()) {
            sb.append("  - ").append(entry.getKey()).append(" (")
              .append(entry.getValue()).append(" books)\n");
        }
        sb.append("\n");
        sb.append("Year range: ").append(minYear).append(" - ")
          .append(maxYear).append("\n");
        sb.append("\n");
        sb.append("Books by availability:\n");
        for (Book book : shelf.books) {
            char marker = book.available ? '+' : '-';
            sb.append("  [").append(marker).append("] ")
              .append(book.title).append(" by ").append(book.author)
              .append(" (").append(book.year).append(") - ")
              .append(book.isbn).append("\n");
        }
        return sb.toString();
    }
}
