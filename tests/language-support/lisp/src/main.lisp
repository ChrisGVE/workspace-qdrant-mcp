;;; Main: entry point and demonstration for the bookshelf system

(defpackage :bookshelf.main
  (:use :common-lisp :bookshelf.models :bookshelf.storage :bookshelf.utils))

(in-package :bookshelf.main)

(defun main ()
  "Run the bookshelf demonstration."
  (let ((shelf (make-shelf :name "Computer Science" :capacity 10))
        (books (list
                (make-book :title "The Art of Computer Programming"
                           :author "Donald Knuth" :year 1968
                           :isbn "9780201896831" :available t)
                (make-book :title "Structure and Interpretation of Computer Programs"
                           :author "Harold Abelson" :year 1996
                           :isbn "9780262510875" :available t)
                (make-book :title "Introduction to Algorithms"
                           :author "Thomas Cormen" :year 2009
                           :isbn "9780262033848" :available nil)
                (make-book :title "Design Patterns"
                           :author "Erich Gamma" :year 1994
                           :isbn "9780201633610" :available t)
                (make-book :title "The Pragmatic Programmer"
                           :author "David Thomas" :year 2019
                           :isbn "9780135957059" :available t))))

    (dolist (book books)
      (add-book shelf book))

    (write-string (generate-report shelf))
    (terpri)
    (terpri)

    (format t "--- Search by author \"knuth\" ---~%")
    (dolist (book (find-by-author shelf "knuth"))
      (format t "  ~A~%" (format-book book)))
    (terpri)

    (format t "--- Search by year range 1990-2010 ---~%")
    (dolist (book (find-by-year-range shelf 1990 2010))
      (format t "  ~A~%" (format-book book)))
    (terpri)

    (format t "--- Parse CSV ---~%")
    (let ((parsed (parse-csv-line "Clean Code,Robert Martin,2008,9780132350884,true")))
      (format t "  Parsed: ~A~%" (format-book parsed)))
    (terpri)

    (format t "--- ISBN Validation ---~%")
    (dolist (book books)
      (let ((status (if (validate-isbn (book-isbn book)) "valid" "invalid")))
        (format t "  ~A: ~A~%" (book-isbn book) status)))))

(main)
