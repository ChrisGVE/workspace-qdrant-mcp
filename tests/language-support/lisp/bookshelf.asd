(defsystem "bookshelf"
  :description "A small library management system"
  :version "1.0.0"
  :serial t
  :components ((:file "src/models")
               (:file "src/storage")
               (:file "src/utils")
               (:file "src/main")))
