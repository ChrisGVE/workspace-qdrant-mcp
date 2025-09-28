# Configuration System Architecture Requirements - VERBATIM USER SPECIFICATION

**Date**: 2025-09-27 22:00
**Status**: CRITICAL - User-specified architecture to implement

## User's Exact Requirements

The user provided these EXACT requirements for the configuration system architecture:

"I think you are seeing this backward, what needs to be done is a) the full yaml file is parsed into a temporary structure (dictionary) unit conversions are done at this point b) an internal dictionary is created with ALL the possible configuration labels and default or NULL values are attributed to them c) both dictionaries are merged with the values of the parsed yaml file taking precedence over the default values both starting dictionaries are droped after the merge and only the result is kept d) this is a global structure available to the full code (read/only), an accessor must be provided to fetch values in the format level1.level2.level3 (if the value is a dictionary that's what is returned, if it's an array an array is returned, if it's a string, a float or an int, this is what is returned) e) everywhere in the code where a configuration value is needed, it will use the global accessor to fetch its value. f) When this works, and only when this works, we can do configuration type checking at the point of parsing, and see whether we have opportunities for graceful fallback."

## Key Architecture Points

1. **Parse YAML into temporary dictionary structure with unit conversions**
2. **Create internal dictionary with ALL possible config labels and defaults/NULL values**
3. **Merge dictionaries (YAML values take precedence over defaults)**
4. **Drop both starting dictionaries, keep only merged result**
5. **Global read-only structure available to full codebase**
6. **Accessor pattern: level1.level2.level3 with type-appropriate returns**
7. **All code uses global accessor for config values**
8. **Type checking and graceful fallback only AFTER this works**

## Implementation Strategy

This completely replaces the current struct-based deserialization approach with a dictionary-based merge system.

**Current Problem**: The struct-based deserialization is failing with "missing field `server`" errors because it expects exact field matches.

**New Solution**: Dictionary-based parsing that is tolerant of missing fields and provides defaults for everything.

## Action Required

Implement this architecture to resolve the configuration parsing failures that are blocking daemon startup.