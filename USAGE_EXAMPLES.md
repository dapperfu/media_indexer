# Media Indexer - Usage Examples

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Basic Usage

### Command Line Interface

```bash
# Process images in a directory
media-indexer /path/to/images

# Specify output directory
media-indexer /path/to/images -o /path/to/output

# Verbosity levels
media-indexer /path/to/images -v      # INFO level
media-indexer /path/to/images -vv    # DETAILED level
media-indexer /path/to/images -vvv   # VERBOSE level
media-indexer /path/to/images -vvvv  # TRACE level (with TQDM progress bars)
media-indexer /path/to/images -vvvvv # DEBUG level

# Resume from checkpoint
media-indexer /path/to/images --resume

# Specify batch size
media-indexer /path/to/images -b 4

# Image format filtering
media-indexer /path/to/images --formats jpg png jpeg

# Database storage (with sidecar files)
media-indexer /path/to/images --db mydb.db

# Database storage only (no sidecar files)
media-indexer /path/to/images --db mydb.db --no-sidecar
```

### Python Library Usage

```python
from media_indexer import ImageProcessor
from pathlib import Path

# Create processor with sidecar only
processor = ImageProcessor(
    input_dir=Path("/path/to/images"),
    output_dir=Path("/path/to/output"),  # optional
    verbose=12,  # TRACE level with TQDM
    batch_size=1,
    checkpoint_file=Path(".checkpoint.json")
)

# Create processor with database storage
processor = ImageProcessor(
    input_dir=Path("/path/to/images"),
    output_dir=Path("/path/to/output"),
    database_path=Path("mydb.db"),
    disable_sidecar=False,  # Also generate sidecar files
    verbose=12,
    batch_size=4,
)

# Create processor with database only (no sidecar files)
processor = ImageProcessor(
    input_dir=Path("/path/to/images"),
    database_path=Path("mydb.db"),
    disable_sidecar=True,  # No sidecar files
    verbose=12,
    batch_size=4,
)

# Process all images
stats = processor.process()

# Check statistics
print(f"Total images: {stats['total_images']}")
print(f"Processed: {stats['processed_images']}")
print(f"Skipped: {stats['skipped_images']}")
print(f"Errors: {stats['error_images']}")
```

### Component Usage

You can also use individual components directly:

```python
from media_indexer.gpu_validator import get_gpu_validator
from media_indexer.exif_extractor import get_exif_extractor
from media_indexer.face_detector import get_face_detector
from media_indexer.object_detector import get_object_detector
from media_indexer.pose_detector import get_pose_detector
from pathlib import Path

# Validate GPU
gpu_validator = get_gpu_validator()
device = gpu_validator.get_device()

# Extract EXIF
exif_extractor = get_exif_extractor()
exif_data = exif_extractor.extract_from_path(Path("image.jpg"))

# Detect faces
face_detector = get_face_detector(device)
faces = face_detector.detect_faces(Path("image.jpg"))

# Detect objects
object_detector = get_object_detector(device)
objects = object_detector.detect_objects(Path("image.jpg"))

# Detect poses
pose_detector = get_pose_detector(device)
poses = pose_detector.detect_poses(Path("image.jpg"))
```

## Development

```bash
# Run linter
make lint

# Format code
make format

# Check formatting
make format-check

# Run tests
make test

# Run all checks
make check
```

## Configuration File (YAML example)

Create `config.yaml`:

```yaml
input_dir: /path/to/images
output_dir: /path/to/output
batch_size: 4
verbose: 15
checkpoint_file: .checkpoint.json
formats:
  - jpg
  - jpeg
  - png
  - tiff
```

Then use:

```bash
media-indexer /path/to/images --config config.yaml
```

## Verbosity Levels Explained

- **Default (no -v)**: INFO level - Shows summary information
- **-v**: DETAILED (17) - Shows file-by-file processing
- **-vv**: VERBOSE (15) - Detailed processing info per file
- **-vvv**: TRACE (12) - Shows progress with TQDM progress bars
- **-vvvv**: DEBUG (10) - Full debugging output

## Sidecar Files

Each image gets a corresponding JSON sidecar file:

```
image1.jpg          -> image1.jpg.json
image2.png          -> image2.png.json
photo.tiff          -> photo.tiff.json
```

The sidecar contains:
- EXIF metadata
- Detected faces with bounding boxes and embeddings
- Detected objects with class labels
- Detected human poses with keypoints

## Database Storage

Media Indexer can store metadata in a SQLite database using PonyORM:

```bash
# Store in database with sidecar files
media-indexer --db mydb.db /path/to/images

# Store in database only (no sidecar files)
media-indexer --db mydb.db --no-sidecar /path/to/images
```

### Database Schema

The database contains the following tables:

- **Image**: Main image metadata (path, hash, size, timestamps)
- **Face**: Detected faces with embeddings and confidence scores
- **Object**: Detected objects with class names and bounding boxes
- **Pose**: Detected human poses with keypoints
- **EXIFData**: EXIF metadata as JSON

### Querying the Database

You can query the database using PonyORM:

```python
from media_indexer.db import Image, Face, Object, Pose, get_db
from pony.orm import db_session

# Get all images
with db_session:
    images = Image.get_all_images()
    for img in images:
        print(f"Image: {img.path}")
        
        # Get faces for this image
        faces = Face.get_by_image(img)
        print(f"  Faces: {len(faces)}")
        
        # Get objects
        objects = Object.get_by_image(img)
        print(f"  Objects: {len(objects)}")
        
        # Get poses
        poses = Pose.get_by_image(img)
        print(f"  Poses: {len(poses)}")

# Query by face confidence
with db_session:
    high_conf_faces = Face.get_by_confidence(0.8)

# Query by object class
with db_session:
    person_objects = Object.get_by_class("person")
```

## Checkpoint/Resume

The tool automatically saves progress to `.checkpoint.json`:

```bash
# Start processing
media-indexer /path/to/images

# If interrupted, resume with:
media-indexer /path/to/images --resume

# Already processed images will be skipped
```

## Error Handling

The tool continues processing even if some images fail:

- Corrupted images are logged and skipped
- Unsupported formats are skipped
- Processing errors are logged but don't stop the run
- Final statistics show total errors

## Performance Tips

1. **Batch Size**: Adjust `-b` based on GPU memory
2. **Multi-GPU**: Automatically uses all available GPUs
3. **Idempotent**: Safe to re-run (skips processed images)
4. **Checkpoint**: Handles interruptions gracefully

## Example Output

```
INFO: REQ-002: Processing images from /path/to/images
INFO: REQ-006: GPU validation successful. Found 2 GPU(s)
INFO: REQ-002: Found 1500 images to process
Processing images: 100%|████████████| 1500/1500 [05:23<00:00,  4.65it/s]
INFO: REQ-012: Processing complete
INFO:   Total images: 1500
INFO:   Processed: 1485
INFO:   Skipped: 10
INFO:   Errors: 5
```

