//! Collection type enumeration and parsing logic.
//!
//! Collection Type Detection:
//! - PROJECT: _{project_id} where project_id is 12-char hex hash
//! - LIBRARY: _{library_name} where library_name is alphanumeric with hyphens
//! - USER: {basename}-{type} format
//! - RULES: exact match "rules" (also accepts legacy "memory")

/// Collection type enumeration for metadata enrichment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectionType {
    /// Project collection (_{project_id})
    Project {
        /// 12-character hex project ID
        project_id: String,
    },
    /// Library collection (_{library_name})
    Library {
        /// Library name (e.g., "fastapi", "pandas")
        library_name: String,
    },
    /// User collection ({basename}-{type})
    User {
        /// Collection basename
        basename: String,
        /// Collection type suffix
        collection_type: String,
    },
    /// Rules collection (exact match "rules", also accepts legacy "memory")
    Rules,
}

impl CollectionType {
    /// Parse collection name to determine collection type
    ///
    /// # Arguments
    /// * `collection_name` - Name of the collection
    ///
    /// # Returns
    /// CollectionType enum variant
    ///
    /// # Examples
    /// ```
    /// use workspace_qdrant_core::metadata_enrichment::CollectionType;
    ///
    /// let ctype = CollectionType::from_name("_0f72d776622e");
    /// assert!(matches!(ctype, CollectionType::Project { .. }));
    ///
    /// let ctype = CollectionType::from_name("_fastapi");
    /// assert!(matches!(ctype, CollectionType::Library { .. }));
    ///
    /// let ctype = CollectionType::from_name("myapp-notes");
    /// assert!(matches!(ctype, CollectionType::User { .. }));
    ///
    /// let ctype = CollectionType::from_name("rules");
    /// assert!(matches!(ctype, CollectionType::Rules));
    /// ```
    pub fn from_name(collection_name: &str) -> Self {
        // Check for "rules" (or legacy "memory") match
        if collection_name == "rules" || collection_name == "memory" {
            return CollectionType::Rules;
        }

        // Check for underscore prefix (PROJECT or LIBRARY)
        if let Some(name_without_underscore) = collection_name.strip_prefix('_') {
            // PROJECT collections are 12-character hex hashes
            if name_without_underscore.len() == 12
                && name_without_underscore.chars().all(|c| c.is_ascii_hexdigit())
            {
                return CollectionType::Project {
                    project_id: name_without_underscore.to_string(),
                };
            }

            // LIBRARY collections are alphanumeric with hyphens/underscores
            return CollectionType::Library {
                library_name: name_without_underscore.to_string(),
            };
        }

        // USER collections have {basename}-{type} format
        if let Some(dash_pos) = collection_name.rfind('-') {
            let basename = collection_name[..dash_pos].to_string();
            let collection_type = collection_name[dash_pos + 1..].to_string();
            return CollectionType::User {
                basename,
                collection_type,
            };
        }

        // Fallback: treat as USER collection with no type suffix
        CollectionType::User {
            basename: collection_name.to_string(),
            collection_type: String::new(),
        }
    }
}
