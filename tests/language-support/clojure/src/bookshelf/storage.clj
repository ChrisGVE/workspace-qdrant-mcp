(ns bookshelf.storage
  (:require [clojure.string :as str]))

(defn is-full
  "Check whether the shelf has reached its capacity."
  [shelf]
  (>= (count (:books shelf)) (:capacity shelf)))

(defn add-book
  "Add a book to the shelf. Returns updated shelf."
  [shelf book]
  (when (is-full shelf)
    (throw (ex-info (str "Shelf '" (:name shelf) "' is at capacity (" (:capacity shelf) ")")
                    {:shelf shelf})))
  (update shelf :books conj book))

(defn remove-book
  "Remove a book by ISBN. Returns updated shelf."
  [shelf isbn]
  (let [books (:books shelf)
        idx   (first (keep-indexed (fn [i b] (when (= (:isbn b) isbn) i)) books))]
    (when (nil? idx)
      (throw (ex-info (str "No book with ISBN '" isbn "' found on shelf") {:isbn isbn})))
    (assoc shelf :books (into (subvec books 0 idx) (subvec books (inc idx))))))

(defn find-by-author
  "Find books by author using case-insensitive substring match."
  [shelf author]
  (let [query (str/lower-case author)]
    (filterv #(str/includes? (str/lower-case (:author %)) query)
             (:books shelf))))

(defn find-by-year-range
  "Find books published within an inclusive year range."
  [shelf start-year end-year]
  (filterv #(and (>= (:year %) start-year) (<= (:year %) end-year))
           (:books shelf)))

(defn sort-by-title
  "Return books sorted by title without mutating the shelf."
  [shelf]
  (sort-by :title (:books shelf)))

(defn generate-report
  "Generate a multi-line formatted report for the shelf."
  [shelf]
  (let [books           (:books shelf)
        total           (count books)
        available-count (count (filter :available books))
        avail-pct       (if (pos? total) (quot (* available-count 100) total) 0)
        cap-pct         (if (pos? (:capacity shelf)) (quot (* total 100) (:capacity shelf)) 0)
        author-order    (distinct (map :author books))
        author-counts   (frequencies (map :author books))
        unique-authors  (count author-order)
        years           (map :year books)
        min-year        (if (seq years) (apply min years) 0)
        max-year        (if (seq years) (apply max years) 0)]
    (str/join
     "\n"
     (concat
      [(str "=== Library Report: " (:name shelf) " ===")
       (str "Total books: " total)
       (str "Available: " available-count " / " total " (" avail-pct "%)")
       (str "Capacity: " total " / " (:capacity shelf) " (" cap-pct "% full)")
       ""
       (str "Authors (" unique-authors " unique):")]
      (map (fn [author]
             (str "  - " author " (" (get author-counts author) " books)"))
           author-order)
      [""
       (str "Year range: " min-year " - " max-year)
       ""
       "Books by availability:"]
      (map (fn [book]
             (str "  [" (if (:available book) "+" "-") "] "
                  (:title book) " by " (:author book)
                  " (" (:year book) ") - " (:isbn book)))
           books)))))
