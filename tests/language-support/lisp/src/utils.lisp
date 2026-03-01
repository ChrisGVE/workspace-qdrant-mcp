;;; Utils: ISBN validation, book formatting, and CSV parsing

(defpackage :bookshelf.utils
  (:use :common-lisp :bookshelf.models)
  (:export #:validate-isbn #:format-book #:parse-csv-line))

(in-package :bookshelf.utils)

(defun validate-isbn (isbn)
  "Validate an ISBN-13 check digit."
  (and (stringp isbn)
       (= 13 (length isbn))
       (every #'digit-char-p isbn)
       (let ((total 0))
         (loop for i from 0 below 13
               for ch across isbn
               for digit = (digit-char-p ch)
               for weight = (if (evenp i) 1 3)
               do (incf total (* digit weight)))
         (zerop (mod total 10)))))

(defun format-book (book)
  "Format a book as a single descriptive line."
  (format nil "\"~A\" by ~A (~A) [ISBN: ~A]"
          (book-title book) (book-author book)
          (book-year book) (book-isbn book)))

(defun split-string (string separator)
  "Split a string by a separator character."
  (loop for start = 0 then (1+ pos)
        for pos = (position separator string :start start)
        collect (subseq string start (or pos (length string)))
        while pos))

(defun trim-whitespace (str)
  "Remove leading and trailing whitespace from a string."
  (string-trim '(#\Space #\Tab #\Newline #\Return) str))

(defun parse-csv-line (line)
  "Parse a CSV line into a book."
  (let ((parts (split-string line #\,)))
    (unless (= 5 (length parts))
      (error "Expected 5 fields, got ~A" (length parts)))
    (let* ((title (trim-whitespace (nth 0 parts)))
           (author (trim-whitespace (nth 1 parts)))
           (year-str (trim-whitespace (nth 2 parts)))
           (isbn (trim-whitespace (nth 3 parts)))
           (avail-str (trim-whitespace (nth 4 parts)))
           (year (handler-case (parse-integer year-str)
                   (error () (error "Invalid year: ~A" year-str))))
           (avail-lower (string-downcase avail-str))
           (available (cond
                        ((string= avail-lower "true") t)
                        ((string= avail-lower "false") nil)
                        (t (error "Invalid boolean: ~A" avail-str)))))
      (make-book :title title :author author :year year
                 :isbn isbn :available available))))
