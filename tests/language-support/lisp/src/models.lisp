;;; Models: data structures for the bookshelf library management system

(defpackage :bookshelf.models
  (:use :common-lisp)
  (:export #:make-book #:book-title #:book-author #:book-year
           #:book-isbn #:book-available
           #:make-shelf #:shelf-name #:shelf-capacity #:shelf-books
           #:shelf-add-book-to-list
           #:+genre-fiction+ #:+genre-non-fiction+ #:+genre-science+
           #:+genre-history+ #:+genre-biography+ #:+genre-technology+
           #:+genre-philosophy+))

(in-package :bookshelf.models)

;; Genre constants
(defparameter +genre-fiction+ "Fiction")
(defparameter +genre-non-fiction+ "NonFiction")
(defparameter +genre-science+ "Science")
(defparameter +genre-history+ "History")
(defparameter +genre-biography+ "Biography")
(defparameter +genre-technology+ "Technology")
(defparameter +genre-philosophy+ "Philosophy")

(defstruct book
  (title "" :type string)
  (author "" :type string)
  (year 0 :type integer)
  (isbn "" :type string)
  (available t :type boolean))

(defstruct shelf
  (name "" :type string)
  (capacity 0 :type integer)
  (books nil :type list))

(defun shelf-add-book-to-list (shelf book)
  "Add a book to the shelf's books list (mutates)."
  (setf (shelf-books shelf)
        (append (shelf-books shelf) (list book))))
