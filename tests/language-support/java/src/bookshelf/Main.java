package bookshelf;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        Shelf shelf = new Shelf("Computer Science", 10);

        Book[] books = {
            new Book("The Art of Computer Programming",
                     "Donald Knuth", 1968, "9780201896831", true),
            new Book("Structure and Interpretation of Computer Programs",
                     "Harold Abelson", 1996, "9780262510875", true),
            new Book("Introduction to Algorithms",
                     "Thomas Cormen", 2009, "9780262033848", false),
            new Book("Design Patterns",
                     "Erich Gamma", 1994, "9780201633610", true),
            new Book("The Pragmatic Programmer",
                     "David Thomas", 2019, "9780135957059", true),
        };

        for (Book book : books) {
            Storage.addBook(shelf, book);
        }

        String report = Storage.generateReport(shelf);
        /* Remove trailing newline since print adds one */
        if (report.endsWith("\n")) {
            report = report.substring(0, report.length() - 1);
        }
        System.out.println(report);

        System.out.println();
        System.out.println("--- Search by author \"knuth\" ---");
        for (Book book : Storage.findByAuthor(shelf, "knuth")) {
            System.out.println("  " + Utils.formatBook(book));
        }
        System.out.println();

        System.out.println("--- Search by year range 1990-2010 ---");
        for (Book book : Storage.findByYearRange(shelf, 1990, 2010)) {
            System.out.println("  " + Utils.formatBook(book));
        }
        System.out.println();

        System.out.println("--- Parse CSV ---");
        Book parsed = Utils.parseCsvLine(
            "Clean Code,Robert Martin,2008,9780132350884,true");
        System.out.println("  Parsed: " + Utils.formatBook(parsed));
        System.out.println();

        System.out.println("--- ISBN Validation ---");
        for (Book book : books) {
            String status = Utils.validateIsbn(book.isbn)
                ? "valid" : "invalid";
            System.out.println("  " + book.isbn + ": " + status);
        }
    }
}
