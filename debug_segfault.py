#!/usr/bin/env python3
"""Debug script to run media-indexer analyze with enhanced error handling and tracing.

This script enables faulthandler to get Python tracebacks on segfaults,
and provides better error context for debugging native code crashes.
"""

import faulthandler
import sys
import os
from pathlib import Path

# Enable faulthandler to get Python tracebacks on segfaults
faulthandler.enable()

# Write traceback to stderr on segfault
faulthandler.dump_traceback_later(timeout=60, exit=True)

# Set environment variable to enable Python debug mode
os.environ["PYTHONFAULTHANDLER"] = "1"

# Add signal handlers for better debugging
import signal

def signal_handler(sig, frame):
    """Handle signals and dump traceback."""
    print(f"\n\n=== Signal {sig} received ===", file=sys.stderr)
    faulthandler.dump_traceback(file=sys.stderr)
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGABRT, signal_handler)
signal.signal(signal.SIGBUS, signal_handler)
signal.signal(signal.SIGFPE, signal_handler)
signal.signal(signal.SIGILL, signal_handler)

# Import and run the CLI
if __name__ == "__main__":
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Add src to path if needed
    src_dir = project_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Import and run the CLI main function
    from media_indexer.cli import main
    
    try:
        main()
    except Exception as e:
        print(f"\n\n=== Exception caught ===", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        faulthandler.dump_traceback(file=sys.stderr)
        sys.exit(1)

