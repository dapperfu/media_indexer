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

