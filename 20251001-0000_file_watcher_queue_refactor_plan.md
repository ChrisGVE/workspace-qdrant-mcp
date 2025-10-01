[38;2;127;132;156m   1[0m [38;2;205;214;244m# File Watcher Queue Refactor Plan[0m
[38;2;127;132;156m   2[0m 
[38;2;127;132;156m   3[0m [38;2;205;214;244m## Current Implementation Analysis[0m
[38;2;127;132;156m   4[0m 
[38;2;127;132;156m   5[0m [38;2;205;214;244m1. **FileWatcher.__init__** accepts:[0m
[38;2;127;132;156m   6[0m [38;2;205;214;244m   - `ingestion_callback: Callable[[str, str], None]` - callback for file ingestion (file_path, collection)[0m
[38;2;127;132;156m   7[0m [38;2;205;214;244m   - Stores callback and calls it in `_trigger_ingestion()`[0m
[38;2;127;132;156m   8[0m 
[38;2;127;132;156m   9[0m [38;2;205;214;244m2. **_trigger_ingestion()** method:[0m
[38;2;127;132;156m  10[0m [38;2;205;214;244m   - Checks if callback is coroutine or regular function[0m
[38;2;127;132;156m  11[0m [38;2;205;214;244m   - Calls callback with (file_path, collection)[0m
[38;2;127;132;156m  12[0m [38;2;205;214;244m   - Has error handling[0m
[38;2;127;132;156m  13[0m 
[38;2;127;132;156m  14[0m [38;2;205;214;244m3. **WatchManager**:[0m
[38;2;127;132;156m  15[0m [38;2;205;214;244m   - Stores `ingestion_callback` and passes it to FileWatcher instances[0m
[38;2;127;132;156m  16[0m [38;2;205;214;244m   - Used in `_start_watcher()` when creating FileWatcher[0m
[38;2;127;132;156m  17[0m 
[38;2;127;132;156m  18[0m [38;2;205;214;244m## Target Implementation[0m
[38;2;127;132;156m  19[0m 
[38;2;127;132;156m  20[0m [38;2;205;214;244mReplace callback pattern with state_manager.enqueue() calls:[0m
[38;2;127;132;156m  21[0m 
[38;2;127;132;156m  22[0m [38;2;205;214;244m### 1. Update FileWatcher.__init__ signature[0m
[38;2;127;132;156m  23[0m [38;2;205;214;244m- Remove `ingestion_callback` parameter[0m
[38;2;127;132;156m  24[0m [38;2;205;214;244m- Add `state_manager: SQLiteStateManager` parameter[0m
[38;2;127;132;156m  25[0m [38;2;205;214;244m- Store state_manager instead of callback[0m
[38;2;127;132;156m  26[0m [38;2;205;214;244m- **Keep event_callback** - it's for watch events, not ingestion[0m
[38;2;127;132;156m  27[0m 
[38;2;127;132;156m  28[0m [38;2;205;214;244m### 2. Update _trigger_ingestion() method[0m
[38;2;127;132;156m  29[0m [38;2;205;214;244m- Remove callback invocation[0m
[38;2;127;132;156m  30[0m [38;2;205;214;244m- Add calls to state_manager methods:[0m
[38;2;127;132;156m  31[0m [38;2;205;214;244m  * `calculate_tenant_id(project_root)` - needs project root detection[0m
[38;2;127;132;156m  32[0m [38;2;205;214;244m  * `get_current_branch(project_root)` - needs project root detection[0m
[38;2;127;132;156m  33[0m [38;2;205;214;244m  * `enqueue(file_path, collection, priority, tenant_id, branch, metadata)`[0m
[38;2;127;132;156m  34[0m [38;2;205;214;244m- Calculate priority: default to 5 (NORMAL)[0m
[38;2;127;132;156m  35[0m [38;2;205;214;244m- Build metadata with event information (change type, timestamp)[0m
[38;2;127;132;156m  36[0m [38;2;205;214;244m- Add proper error handling and logging[0m
[38;2;127;132;156m  37[0m 
[38;2;127;132;156m  38[0m [38;2;205;214;244m### 3. Project Root Detection[0m
[38;2;127;132;156m  39[0m [38;2;205;214;244mNeed to detect project root from file_path:[0m
[38;2;127;132;156m  40[0m [38;2;205;214;244m- Walk up directory tree looking for .git directory[0m
[38;2;127;132;156m  41[0m [38;2;205;214;244m- If no .git found, use file's parent directory as fallback[0m
[38;2;127;132;156m  42[0m [38;2;205;214;244m- Cache project root per watch path for performance[0m
[38;2;127;132;156m  43[0m 
[38;2;127;132;156m  44[0m [38;2;205;214;244m### 4. WatchManager updates[0m
[38;2;127;132;156m  45[0m [38;2;205;214;244m- Replace `ingestion_callback` with `state_manager`[0m
[38;2;127;132;156m  46[0m [38;2;205;214;244m- Update `set_ingestion_callback()` to `set_state_manager()`[0m
[38;2;127;132;156m  47[0m [38;2;205;214;244m- Update `_start_watcher()` to pass state_manager instead of callback[0m
[38;2;127;132;156m  48[0m [38;2;205;214;244m- Update initialization check to use state_manager[0m
[38;2;127;132;156m  49[0m 
[38;2;127;132;156m  50[0m [38;2;205;214;244m### 5. Maintain existing functionality[0m
[38;2;127;132;156m  51[0m [38;2;205;214;244m- Keep all filtering logic (language-aware, patterns, ignore patterns)[0m
[38;2;127;132;156m  52[0m [38;2;205;214;244m- Keep debouncing mechanism[0m
[38;2;127;132;156m  53[0m [38;2;205;214;244m- Keep async/await pattern[0m
[38;2;127;132;156m  54[0m [38;2;205;214;244m- Keep event callback system (for watch events)[0m
[38;2;127;132;156m  55[0m 
[38;2;127;132;156m  56[0m [38;2;205;214;244m## Implementation Steps[0m
[38;2;127;132;156m  57[0m 
[38;2;127;132;156m  58[0m [38;2;205;214;244m1. Add project root detection helper method to FileWatcher[0m
[38;2;127;132;156m  59[0m [38;2;205;214;244m2. Update FileWatcher.__init__ to accept state_manager[0m
[38;2;127;132;156m  60[0m [38;2;205;214;244m3. Rewrite _trigger_ingestion() to use queue operations[0m
[38;2;127;132;156m  61[0m [38;2;205;214;244m4. Update WatchManager to use state_manager[0m
[38;2;127;132;156m  62[0m [38;2;205;214;244m5. Write comprehensive unit tests[0m
[38;2;127;132;156m  63[0m 
[38;2;127;132;156m  64[0m [38;2;205;214;244m## Test Coverage[0m
[38;2;127;132;156m  65[0m 
[38;2;127;132;156m  66[0m [38;2;205;214;244mTests should verify:[0m
[38;2;127;132;156m  67[0m [38;2;205;214;244m1. FileWatcher accepts state_manager instead of callback[0m
[38;2;127;132;156m  68[0m [38;2;205;214;244m2. _trigger_ingestion() calls state_manager.enqueue() with correct parameters[0m
[38;2;127;132;156m  69[0m [38;2;205;214;244m3. Project root detection works correctly[0m
[38;2;127;132;156m  70[0m [38;2;205;214;244m4. Tenant ID and branch are calculated properly[0m
[38;2;127;132;156m  71[0m [38;2;205;214;244m5. Priority defaults to 5[0m
[38;2;127;132;156m  72[0m [38;2;205;214;244m6. Metadata includes event information[0m
[38;2;127;132;156m  73[0m [38;2;205;214;244m7. Error handling works correctly[0m
[38;2;127;132;156m  74[0m [38;2;205;214;244m8. Debouncing still functions[0m
[38;2;127;132;156m  75[0m [38;2;205;214;244m9. Filtering logic remains intact[0m
