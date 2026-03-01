;;; Storage: business logic for managing books on shelves

(defpackage :bookshelf.storage
  (:use :common-lisp :bookshelf.models)
  (:export #:add-book #:remove-book #:find-by-author
           #:find-by-year-range #:sort-by-title #:is-full
           #:generate-report))

(in-package :bookshelf.storage)

(defun is-full (shelf)
  "Check whether the shelf has reached its capacity."
  (>= (length (shelf-books shelf)) (shelf-capacity shelf)))

(defun add-book (shelf book)
  "Add a book to the shelf."
  (when (is-full shelf)
    (error "Shelf '~A' is at capacity (~A)"
           (shelf-name shelf) (shelf-capacity shelf)))
  (shelf-add-book-to-list shelf book))

(defun remove-book (shelf isbn)
  "Remove a book by ISBN from the shelf."
  (let ((found nil))
    (setf (shelf-books shelf)
          (loop for book in (shelf-books shelf)
                if (and (not found) (string= (book-isbn book) isbn))
                  do (setf found book)
                else
                  collect book))
    (unless found
      (error "No book with ISBN '~A' found on shelf" isbn))
    found))

(defun find-by-author (shelf author)
  "Find books by author using case-insensitive substring match."
  (let ((query (string-downcase author)))
    (remove-if-not
     (lambda (book)
       (search query (string-downcase (book-author book))))
     (shelf-books shelf))))

(defun find-by-year-range (shelf start-year end-year)
  "Find books published within an inclusive year range."
  (remove-if-not
   (lambda (book)
     (and (>= (book-year book) start-year)
          (<= (book-year book) end-year)))
   (shelf-books shelf)))

(defun sort-by-title (shelf)
  "Return books sorted by title without mutating the shelf."
  (sort (copy-list (shelf-books shelf))
        #'string< :key #'book-title))

(defun generate-report (shelf)
  "Generate a multi-line formatted report for the shelf."
  (let* ((books (shelf-books shelf))
         (total (length books))
         (available-count (count-if #'book-available books))
         (avail-pct (if (> total 0) (floor (* available-count 100) total) 0))
         (cap-pct (if (> (shelf-capacity shelf) 0)
                      (floor (* total 100) (shelf-capacity shelf))
                      0))
         (author-order nil)
         (author-counts (make-hash-table :test #'equal))
         (years (mapcar #'book-year books))
         (min-year (if years (apply #'min years) 0))
         (max-year (if years (apply #'max years) 0)))
    ;; Build author counts preserving insertion order
    (dolist (book books)
      (let ((author (book-author book)))
        (unless (gethash author author-counts)
          (push author author-order))
        (incf (gethash author author-counts 0))))
    (setf author-order (nreverse author-order))
    (let ((unique-authors (length author-order)))
      (with-output-to-string (s)
        (format s "=== Library Report: ~A ===~%" (shelf-name shelf))
        (format s "Total books: ~A~%" total)
        (format s "Available: ~A / ~A (~A%)~%" available-count total avail-pct)
        (format s "Capacity: ~A / ~A (~A% full)~%" total (shelf-capacity shelf) cap-pct)
        (format s "~%")
        (format s "Authors (~A unique):~%" unique-authors)
        (dolist (author author-order)
          (format s "  - ~A (~A books)~%" author (gethash author author-counts)))
        (format s "~%")
        (format s "Year range: ~A - ~A~%" min-year max-year)
        (format s "~%")
        (format s "Books by availability:~%")
        (loop for book in books
              for i from 0
              do (let ((marker (if (book-available book) "+" "-")))
                   (if (= i (1- total))
                       (format s "  [~A] ~A by ~A (~A) - ~A"
                               marker (book-title book) (book-author book)
                               (book-year book) (book-isbn book))
                       (format s "  [~A] ~A by ~A (~A) - ~A~%"
                               marker (book-title book) (book-author book)
                               (book-year book) (book-isbn book)))))))))
