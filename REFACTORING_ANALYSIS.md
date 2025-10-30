# Refactoring Analysis for Media Indexer

## Executive Summary

This document analyzes the codebase for refactoring opportunities to achieve:
- Files ≤ 500 lines
- Line length ≤ 120 characters
- Eliminate duplicate code
- Maximize abstraction opportunities

**Date:** 2024-12-19  
**Total Lines:** ~5,558 lines across 27 Python files

---

## 1. File Size Analysis

### Files Exceeding 500 Lines

| File | Current Lines | Target | Reduction Needed |
|------|--------------|--------|------------------|
| `processor.py` | 1,142 | ≤500 | ~642 lines |
| `cli.py` | 661 | ≤500 | ~161 lines |
| `sidecar_converter.py` | 573 | ≤500 | ~73 lines |

### Files Within Limits (< 500 lines)
- `benchmark_sidecar_scanning.py`: 306 lines ✓
- `raw_converter.py`: 286 lines ✓
- `face_detector.py`: 282 lines ✓
- `feature_extractor.py`: 218 lines ✓
- `db_manager.py`: 217 lines ✓
- All other files: < 200 lines ✓

---

## 2. Line Length Analysis

### Current State
- Maximum line length: **150 characters** (in `cli.py`)
- Most files: 100-120 characters average
- Some long lines exist in:
  - `cli.py`: Help text lines (~150 chars)
  - `processor.py`: Some logging statements (~121 chars)
  - `sidecar_converter.py`: Some function calls (~115 chars)

### Recommendation
- Most files are already close to 120-character limit
- Only minor adjustments needed for help text and long comments
- **Action:** Wrap long strings and reformat docstrings

---

## 3. Duplicate Code Patterns Identified

### 3.1 Signal Handling & Shutdown Management
**Location:** `processor.py` (lines 59-70) and `sidecar_converter.py` (lines 169-180)

**Duplicate Code:**
```python
# Pattern repeated 2x
def _signal_handler(signum: int, frame: Any) -> None:
    logger.warning("REQ-015: Processing interrupted by user")
    _shutdown_flag.set()
```

**Impact:** 2 instances, ~12 lines each

**Recommendation:** Create `utils/signal_handler.py` with reusable `CancellationManager` class

### 3.2 ThreadPoolExecutor Cancellation Pattern
**Location:** `processor.py` (lines 857-898, 982-1094) and `sidecar_converter.py` (lines 244-318, 492-567)

**Duplicate Pattern:**
```python
# Pattern repeated 4x with minor variations
executor: ThreadPoolExecutor | None = None
try:
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(...): item for item in items}
    for future in as_completed(futures):
        if _shutdown_flag.is_set():
            # Cancel remaining futures
            for f in futures:
                if not f.done():
                    f.cancel()
            break
        # ... process results
except KeyboardInterrupt:
    _shutdown_flag.set()
finally:
    if executor:
        if _shutdown_flag.is_set():
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)
```

**Impact:** 4 instances, ~40-50 lines each = ~180 lines of duplicate code

**Recommendation:** Create `utils/parallel_executor.py` with `ParallelExecutor` context manager

### 3.3 Database Metadata Conversion
**Location:** `processor.py` (lines 585-635) and `sidecar_converter.py` (lines 109-157)

**Duplicate Pattern:**
```python
# Pattern repeated 2x
if "faces" in metadata:
    for face_data in metadata["faces"]:
        face_kwargs = {
            "image": db_image,
            "confidence": face_data.get("confidence", 0.0),
            "bbox": face_data.get("bbox", []),
            "model": face_data.get("model", "unknown"),
        }
        embedding = face_data.get("embedding")
        if embedding is not None:
            face_kwargs["embedding"] = embedding
        Face(**face_kwargs)
# Similar for objects, poses, exif
```

**Impact:** 2 instances, ~90 lines each = ~180 lines of duplicate code

**Recommendation:** Create `db/metadata_converter.py` with conversion utilities

### 3.4 Metadata Building from Database Entities
**Location:** `sidecar_converter.py` (lines 370-413)

**Duplicate Pattern:**
```python
# Pattern similar to conversion above
metadata = {"faces": [], "objects": [], "poses": [], "exif": None}
for face in db_image.faces:
    metadata["faces"].append({...})
for obj in db_image.objects:
    metadata["objects"].append({...})
# etc.
```

**Impact:** 1 instance, but similar logic exists = ~50 lines

**Recommendation:** Extend `db/metadata_converter.py` to handle both directions

### 3.5 Progress Bar Creation
**Location:** Multiple files using `create_progress_bar_with_global_speed`

**Status:** Already abstracted ✓

### 3.6 Sidecar File Reading
**Location:** `processor.py` (lines 656-667, 798-803) and `sidecar_converter.py` (lines 73-74)

**Duplicate Pattern:**
```python
# Pattern repeated 3x
if self.sidecar_generator:
    metadata = self.sidecar_generator.read_sidecar(sidecar_path)
else:
    with open(sidecar_path) as f:
        metadata = json.load(f)
```

**Impact:** 3 instances, ~6 lines each = ~18 lines

**Recommendation:** Add utility function in `utils/file_utils.py` or `sidecar_generator.py`

### 3.7 Database Session Management
**Location:** Multiple files (`db_manager.py`, `feature_extractor.py`, `sidecar_converter.py`)

**Pattern:**
```python
# Pattern repeated 3x
db_conn = DatabaseConnection(database_path)
db_conn.connect()
try:
    with db_session:
        # ... operations
finally:
    db_conn.close()
```

**Impact:** 3 instances, ~6 lines each = ~18 lines

**Recommendation:** Create `db/session_manager.py` with context manager

---

## 4. Refactoring Recommendations

### 4.1 Split `processor.py` (1,142 → ~500 lines)

**Current Structure:**
- `_signal_handler()`: 12 lines
- Progress bar classes: ~80 lines
- `create_rich_progress_bar()`: ~70 lines
- `ImageProcessor.__init__()`: ~80 lines
- Helper methods: ~200 lines
- `process()`: ~380 lines
- `_process_single_image()`: ~92 lines

**Proposed Split:**

1. **`processor/__init__.py`** (20 lines)
   - Exports main `ImageProcessor` class

2. **`processor/core.py`** (~400 lines)
   - `ImageProcessor` class (main logic)
   - `__init__`, `_initialize_components`, `_get_image_files`
   - `_needs_processing`, `_get_required_analyses`
   - `process()` method (orchestration)

3. **`processor/image_processor.py`** (~250 lines)
   - `_process_single_image()`
   - `_store_to_database()`
   - Image processing logic

4. **`processor/analysis_scanner.py`** (~200 lines)
   - `_find_sidecar_path()`
   - `_get_existing_analyses()`
   - `_get_existing_analyses_from_database_batch()`
   - Sidecar/database scanning logic

5. **`processor/progress.py`** (~150 lines)
   - `create_rich_progress_bar()`
   - `AvgSpeedColumn`, `CurrentFileColumn`, `DetectionsColumn`
   - Progress-related utilities

6. **`processor/statistics.py`** (~50 lines)
   - `_update_stats_increment()`
   - `_print_statistics()`
   - Statistics tracking

### 4.2 Split `cli.py` (661 → ~500 lines)

**Current Structure:**
- `setup_logging()`: ~45 lines
- `add_common_args()`: ~42 lines
- `parse_args()`: ~240 lines
- Subcommand handlers: ~300 lines
- `main()`: ~37 lines

**Proposed Split:**

1. **`cli/__init__.py`** (20 lines)
   - Exports `main()` function

2. **`cli/main.py`** (~100 lines)
   - `main()` function
   - Routing logic

3. **`cli/parsing.py`** (~250 lines)
   - `parse_args()`
   - `setup_logging()`
   - `add_common_args()`
   - Argument parsing logic

4. **`cli/commands/extract.py`** (~80 lines)
   - `process_extract()` function

5. **`cli/commands/analyze.py`** (~80 lines)
   - `process_analyze()` function

6. **`cli/commands/convert.py`** (~70 lines)
   - `process_convert()` function

7. **`cli/commands/db.py`** (~100 lines)
   - `process_db()` function

### 4.3 Split `sidecar_converter.py` (573 → ~500 lines)

**Current Structure:**
- `_import_single_sidecar()`: ~125 lines
- `import_sidecars_to_database()`: ~140 lines
- `_export_single_image()`: ~100 lines
- `export_database_to_sidecars()`: ~140 lines
- Signal handling: ~12 lines

**Proposed Split:**

1. **`sidecar_converter/__init__.py`** (20 lines)
   - Exports main functions

2. **`sidecar_converter/importer.py`** (~200 lines)
   - `_import_single_sidecar()`
   - `import_sidecars_to_database()`
   - Uses shared cancellation manager

3. **`sidecar_converter/exporter.py`** (~200 lines)
   - `_export_single_image()`
   - `export_database_to_sidecars()`
   - Uses shared cancellation manager

### 4.4 Create Shared Utilities

#### 4.4.1 `utils/cancellation.py` (~80 lines)
```python
"""
Cancellation manager for graceful shutdown.
"""
class CancellationManager:
    """Thread-safe cancellation manager."""
    def __init__(self):
        self._flag = threading.Event()
        self._original_handler = None
    
    def setup_signal_handler(self):
        """Register signal handler."""
        
    def reset(self):
        """Reset cancellation flag."""
        
    def is_cancelled(self) -> bool:
        """Check if cancelled."""
        
    def cancel(self):
        """Set cancellation flag."""
        
    def cleanup(self):
        """Restore original signal handler."""
```

#### 4.4.2 `utils/parallel_executor.py` (~120 lines)
```python
"""
Parallel executor with cancellation support.
"""
class ParallelExecutor:
    """Context manager for parallel execution with cancellation."""
    def __init__(
        self,
        workers: int,
        cancellation_manager: CancellationManager,
        progress_bar: Any | None = None,
    ):
        ...
    
    def execute(
        self,
        items: list[T],
        func: Callable[[T], tuple[bool, Any]],
    ) -> list[tuple[bool, Any]]:
        """Execute function on items in parallel."""
        ...
```

#### 4.4.3 `db/metadata_converter.py` (~200 lines)
```python
"""
Convert between metadata dicts and database entities.
"""
class MetadataConverter:
    """Convert metadata between dict and database formats."""
    
    @staticmethod
    def metadata_to_db_entities(
        db_image: Image,
        metadata: dict[str, Any],
    ) -> None:
        """Store metadata dict as database entities."""
        ...
    
    @staticmethod
    def db_entities_to_metadata(
        db_image: Image,
    ) -> dict[str, Any]:
        """Convert database entities to metadata dict."""
        ...
```

#### 4.4.4 `db/session_manager.py` (~60 lines)
```python
"""
Database session context manager.
"""
class DatabaseSession:
    """Context manager for database sessions."""
    def __init__(self, database_path: Path):
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, ...):
        ...
```

#### 4.4.5 `utils/sidecar_utils.py` (~40 lines)
```python
"""
Sidecar file utilities.
"""
def read_sidecar_metadata(
    sidecar_path: Path,
    sidecar_generator: SidecarGenerator | None = None,
) -> dict[str, Any]:
    """Read sidecar metadata with fallback."""
    ...
```

---

## 5. Abstraction Opportunities

### 5.1 High-Level Abstractions

#### 5.1.1 Processing Pipeline
Create a generic pipeline pattern:
```python
class ProcessingPipeline:
    """Generic processing pipeline."""
    def __init__(self, stages: list[PipelineStage]):
        ...
    
    def execute(self, items: list[T]) -> list[Result]:
        """Execute pipeline on items."""
        ...
```

#### 5.1.2 Metadata Storage Strategy
Abstract storage (sidecar vs database):
```python
class MetadataStorageStrategy(ABC):
    """Abstract storage strategy."""
    @abstractmethod
    def store(self, image_path: Path, metadata: dict) -> bool:
        ...
    
    @abstractmethod
    def retrieve(self, image_path: Path) -> dict | None:
        ...

class SidecarStorageStrategy(MetadataStorageStrategy):
    ...

class DatabaseStorageStrategy(MetadataStorageStrategy):
    ...
```

#### 5.1.3 Analysis Orchestrator
Abstract analysis execution:
```python
class AnalysisOrchestrator:
    """Orchestrate image analyses."""
    def __init__(self, detectors: dict[str, Detector]):
        ...
    
    def analyze(
        self,
        image_path: Path,
        required_analyses: set[str],
    ) -> dict[str, Any]:
        """Run required analyses."""
        ...
```

### 5.2 Medium-Level Abstractions

#### 5.2.1 Detector Interface
Standardize detector interface:
```python
class Detector(ABC):
    """Base detector interface."""
    @abstractmethod
    def detect(self, image_path: Path) -> list[Detection]:
        ...
    
    @abstractmethod
    def get_name(self) -> str:
        ...
```

#### 5.2.2 Progress Tracking
Abstract progress tracking:
```python
class ProgressTracker(ABC):
    """Abstract progress tracker."""
    @abstractmethod
    def start(self, total: int, desc: str) -> None:
        ...
    
    @abstractmethod
    def update(self, n: int = 1, **kwargs) -> None:
        ...
    
    @abstractmethod
    def finish(self) -> None:
        ...
```

### 5.3 Low-Level Abstractions

#### 5.3.1 File Operations
Already well abstracted in `utils/file_utils.py` ✓

#### 5.3.2 Image Loading
Could abstract image loading:
```python
class ImageLoader:
    """Unified image loading interface."""
    def load(self, path: Path) -> np.ndarray | Image.Image:
        """Load image with support for RAW."""
        ...
```

---

## 6. Implementation Plan

### Phase 1: Create Shared Utilities (High Impact, Low Risk)
1. Create `utils/cancellation.py`
2. Create `utils/parallel_executor.py`
3. Create `db/metadata_converter.py`
4. Create `db/session_manager.py`
5. Create `utils/sidecar_utils.py`
6. **Estimated reduction:** ~400 lines of duplicate code

### Phase 2: Refactor Large Files (Medium Impact, Medium Risk)
1. Split `processor.py` into `processor/` module
2. Split `cli.py` into `cli/` module
3. Split `sidecar_converter.py` into `sidecar_converter/` module
4. **Estimated reduction:** ~200 lines (better organization)

### Phase 3: Implement Abstractions (Low Impact, High Risk)
1. Implement `MetadataStorageStrategy`
2. Implement `AnalysisOrchestrator`
3. Implement `Detector` interface
4. **Estimated reduction:** Varies (better extensibility)

### Phase 4: Line Length Cleanup (Low Impact, Low Risk)
1. Wrap long lines in help text
2. Format docstrings
3. Split long function calls
4. **Action:** Pass through codebase with formatter

---

## 7. Expected Outcomes

### After Phase 1 + Phase 2:
- **Files ≤ 500 lines:** ✅ All files
- **Duplicate code reduction:** ~400 lines eliminated
- **Line length:** Most files ≤ 120 chars (minor fixes needed)
- **Maintainability:** Significantly improved

### After Phase 3:
- **Extensibility:** Much easier to add new detectors/storage backends
- **Testability:** Better separation of concerns
- **Code reuse:** Higher abstraction levels

---

## 8. Risk Assessment

### Low Risk
- Creating shared utilities (Phase 1)
- Line length cleanup (Phase 4)

### Medium Risk
- Splitting large files (Phase 2)
  - Requires careful import management
  - Need to maintain backward compatibility

### High Risk
- Implementing abstractions (Phase 3)
  - Could introduce breaking changes
  - Requires thorough testing

---

## 9. Recommendations

### Priority 1 (Do First)
1. ✅ Create shared utilities (`cancellation.py`, `parallel_executor.py`)
2. ✅ Create `db/metadata_converter.py` to eliminate duplicate conversion code
3. ✅ Split `processor.py` (largest file, highest impact)

### Priority 2 (Do Next)
4. ✅ Split `cli.py` and `sidecar_converter.py`
5. ✅ Create `db/session_manager.py` for cleaner session handling
6. ✅ Line length cleanup pass

### Priority 3 (Consider Later)
7. ⚠️ Implement `MetadataStorageStrategy` abstraction
8. ⚠️ Implement `AnalysisOrchestrator`
9. ⚠️ Implement `Detector` interface

---

## 10. Summary

### Current State
- 3 files exceed 500 lines
- ~400 lines of duplicate code identified
- Line lengths mostly acceptable (minor fixes needed)

### After Refactoring (Phase 1 + 2)
- ✅ All files ≤ 500 lines
- ✅ ~400 lines of duplicate code eliminated
- ✅ Better code organization
- ✅ Improved maintainability

### Abstraction Potential
- **High:** Processing pipeline, storage strategies
- **Medium:** Detector interface, progress tracking
- **Low:** Already well abstracted (file utils, image utils)

The codebase is well-structured overall. The main refactoring opportunities are:
1. Eliminating duplicate code patterns
2. Splitting large files for better organization
3. Creating shared utilities for common operations

These changes will significantly improve maintainability without introducing major architectural changes.

