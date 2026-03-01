(ns bookshelf.utils
  (:require [bookshelf.models :as models]
            [clojure.string :as str]))

(defn validate-isbn
  "Validate an ISBN-13 check digit."
  [isbn]
  (and (string? isbn)
       (= 13 (count isbn))
       (every? #(Character/isDigit %) isbn)
       (let [digits (map #(Character/getNumericValue %) isbn)
             total  (reduce + (map-indexed
                               (fn [i d] (* d (if (even? i) 1 3)))
                               digits))]
         (zero? (mod total 10)))))

(defn format-book
  "Format a book as a single descriptive line."
  [book]
  (str "\"" (:title book) "\" by " (:author book)
       " (" (:year book) ") [ISBN: " (:isbn book) "]"))

(defn parse-csv-line
  "Parse a CSV line into a book map."
  [line]
  (let [parts (str/split line #",")]
    (when (not= 5 (count parts))
      (throw (ex-info (str "Expected 5 fields, got " (count parts))
                      {:parts parts})))
    (let [[title author year-str isbn avail-str] (map str/trim parts)
          year (try (Integer/parseInt year-str)
                    (catch NumberFormatException _
                      (throw (ex-info (str "Invalid year: " year-str) {}))))
          avail-lower (str/lower-case avail-str)
          available (cond
                      (= avail-lower "true")  true
                      (= avail-lower "false") false
                      :else (throw (ex-info (str "Invalid boolean: " avail-str) {})))]
      (models/make-book title author year isbn available))))
