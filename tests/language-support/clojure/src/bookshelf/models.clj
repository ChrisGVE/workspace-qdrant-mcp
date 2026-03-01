(ns bookshelf.models)

(def genres
  {:fiction     "Fiction"
   :non-fiction "NonFiction"
   :science     "Science"
   :history     "History"
   :biography   "Biography"
   :technology  "Technology"
   :philosophy  "Philosophy"})

(defn make-book
  "Create a book map with the given fields."
  [title author year isbn available]
  {:title     title
   :author    author
   :year      year
   :isbn      isbn
   :available available})

(defn make-shelf
  "Create a shelf map with the given name and capacity."
  [name capacity]
  {:name     name
   :capacity capacity
   :books    []})
