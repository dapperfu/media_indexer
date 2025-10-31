#!/bin/bash
# Debug script to run media-indexer with gdb to catch segfaults

set -e

cd /projects/media_indexer

# Find the Python interpreter from venv
PYTHON="${VIRTUAL_ENV:-venv}/bin/python3"
if [ ! -f "$PYTHON" ]; then
    PYTHON=$(which python3)
fi

echo "Using Python: $PYTHON"
echo "Running: media-indexer analyze --db 2016.db --batch-size 16 --workers 16 /tun/pictures/2016/"
echo ""
echo "Starting gdb session..."
echo "Use 'run' to start, 'bt' for backtrace when it crashes, 'continue' to continue"
echo ""

# Run with gdb
gdb -ex "set args analyze --db 2016.db --batch-size 16 --workers 16 /tun/pictures/2016/" \
    -ex "set environment PYTHONFAULTHANDLER=1" \
    -ex "set print pretty on" \
    -ex "handle SIGSEGV stop print" \
    -ex "handle SIGABRT stop print" \
    -ex "run" \
    -ex "bt" \
    -ex "info registers" \
    -ex "quit" \
    --args "$PYTHON" -m media_indexer.cli

