package bookshelf;

public class Utils {

    public static boolean validateIsbn(String isbn) {
        if (isbn.length() != 13) return false;
        for (char c : isbn.toCharArray()) {
            if (c < '0' || c > '9') return false;
        }
        int total = 0;
        for (int i = 0; i < 13; i++) {
            int digit = isbn.charAt(i) - '0';
            total += (i % 2 == 0) ? digit : digit * 3;
        }
        return total % 10 == 0;
    }

    public static String formatBook(Book book) {
        return "\"" + book.title + "\" by " + book.author
            + " (" + book.year + ") [ISBN: " + book.isbn + "]";
    }

    public static Book parseCsvLine(String line) {
        String[] parts = line.split(",", -1);
        if (parts.length != 5) {
            throw new RuntimeException(
                "Expected 5 fields, got " + parts.length);
        }

        int year;
        try {
            year = Integer.parseInt(parts[2].trim());
        } catch (NumberFormatException e) {
            throw new RuntimeException("Invalid year: " + parts[2]);
        }

        String availStr = parts[4].trim().toLowerCase();
        boolean available;
        if (availStr.equals("true")) {
            available = true;
        } else if (availStr.equals("false")) {
            available = false;
        } else {
            throw new RuntimeException("Invalid boolean: " + parts[4]);
        }

        return new Book(parts[0].trim(), parts[1].trim(), year,
                        parts[3].trim(), available);
    }
}
