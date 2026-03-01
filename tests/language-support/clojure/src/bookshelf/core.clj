(ns bookshelf.core
  (:require [bookshelf.models :as models]
            [bookshelf.storage :as storage]
            [bookshelf.utils :as utils]))

(defn -main
  "Run the bookshelf demonstration."
  [& _args]
  (let [shelf  (models/make-shelf "Computer Science" 10)
        books  [(models/make-book "The Art of Computer Programming" "Donald Knuth" 1968 "9780201896831" true)
                (models/make-book "Structure and Interpretation of Computer Programs" "Harold Abelson" 1996 "9780262510875" true)
                (models/make-book "Introduction to Algorithms" "Thomas Cormen" 2009 "9780262033848" false)
                (models/make-book "Design Patterns" "Erich Gamma" 1994 "9780201633610" true)
                (models/make-book "The Pragmatic Programmer" "David Thomas" 2019 "9780135957059" true)]
        shelf  (reduce storage/add-book shelf books)]

    (println (storage/generate-report shelf))
    (println)

    (println "--- Search by author \"knuth\" ---")
    (doseq [book (storage/find-by-author shelf "knuth")]
      (println (str "  " (utils/format-book book))))
    (println)

    (println "--- Search by year range 1990-2010 ---")
    (doseq [book (storage/find-by-year-range shelf 1990 2010)]
      (println (str "  " (utils/format-book book))))
    (println)

    (println "--- Parse CSV ---")
    (let [parsed (utils/parse-csv-line "Clean Code,Robert Martin,2008,9780132350884,true")]
      (println (str "  Parsed: " (utils/format-book parsed))))
    (println)

    (println "--- ISBN Validation ---")
    (doseq [book books]
      (let [status (if (utils/validate-isbn (:isbn book)) "valid" "invalid")]
        (println (str "  " (:isbn book) ": " status))))))
