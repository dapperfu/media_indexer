# Media Indexer

GPU-accelerated image analysis tool for extracting metadata, faces, objects, and poses from large image collections.

## Features

- **GPU-accelerated processing** - No CPU fallback, GPU-only operation
- **Parallel processing** - Thread-based I/O with batch processing (default 4 images/batch for 12GB VRAM)
- **EXIF extraction** - Fast EXIF parsing using fast-exif-rs-py
- **Face detection** - Multi-model approach using insightface, YOLOv8-face, and YOLOv11-face
- **Object detection** - YOLOv12x for comprehensive object detection
- **Pose detection** - YOLOv11-pose for human pose estimation
- **Sidecar files** - JSON format sidecar files containing extracted metadata
- **Checkpoint/Resume** - Resume interrupted processing
- **Progress tracking** - Real-time progress with TQDM
- **Multi-level verbosity** - Detailed logging control

## Requirements

- Python 3.10+
- CUDA-capable GPU
- NVIDIA GPU drivers

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Download models to central cache (optional, will auto-download on first run)
media-indexer-download-models --download-yolo --download-insightface

# Or just list cached models
media-indexer-download-models --list
```

### Centralized Model Cache

Models are stored in `~/.media_indexer/models/` by default. This prevents re-downloading on every install:

- **YOLO models** (REQ-008, REQ-009): `~/.media_indexer/models/yolo/`
- **InsightFace models** (REQ-007): `~/.media_indexer/models/insightface/`

You can specify a custom cache location:

```bash
media-indexer /path/to/images --cache-dir /custom/path
```

## Usage

### Command Line

```bash
# Basic usage
media-indexer /path/to/images

# With options
media-indexer /path/to/images -o /path/to/output -vvv

# Resume from checkpoint
media-indexer /path/to/images --resume

# Verbosity levels
media-indexer /path/to/images -v      # INFO
media-indexer /path/to/images -vv     # DETAILED
media-indexer /path/to/images -vvv    # VERBOSE
media-indexer /path/to/images -vvvv   # TRACE (with TQDM)
media-indexer /path/to/images -vvvvv # DEBUG
```

### Python Library

```python
from media_indexer import ImageProcessor

processor = ImageProcessor(
    input_dir="/path/to/images",
    output_dir="/path/to/output",
    verbose=12  # TRACE level with TQDM
)

stats = processor.process()
print(stats)
```

## Configuration

Configuration file support (YAML/TOML):

```yaml
input_dir: /path/to/images
output_dir: /path/to/output
batch_size: 4
verbose: 15
checkpoint_file: .checkpoint.json
```

## Requirements Traceability

All code is labeled with requirement IDs (REQ-###) as specified in `requirements.sdoc`.

## License

MIT

## Project Structure

```
media_indexer/
├── requirements.sdoc          # Requirements in StrictDoc format
├── pyproject.toml            # Project configuration
├── src/media_indexer/        # Source code
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── processor.py         # Main processor
│   ├── gpu_validator.py     # GPU validation (REQ-006)
│   ├── exif_extractor.py    # EXIF extraction (REQ-003)
│   ├── face_detector.py     # Face detection (REQ-007)
│   ├── object_detector.py   # Object detection (REQ-008)
│   ├── pose_detector.py     # Pose detection (REQ-009)
│   └── sidecar_generator.py # Sidecar generation (REQ-004)
└── tests/                    # Test suite
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run linters
make lint

# Format code
make format

# Run all checks (lint + test)
make check
```

## Performance

Optimized for processing large image collections (2.2TB+):

- GPU-accelerated inference
- Batch processing with configurable batch size
- Checkpoint/resume for fault tolerance
- Idempotent processing (skip already-processed images)

