"""Automated CUDA-enabled dlib installation.

REQ-073: Provide a scripted workflow that downloads, patches, and installs
dlib with the appropriate CUDA compute capability so GPU features are always
enabled without manual intervention.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable


DEFAULT_DLIB_VERSION = "20.0.0"
CUDA_TEST_FILES = (
    Path("dlib/cmake_utils/test_for_cuda/CMakeLists.txt"),
    Path("dlib/cmake_utils/test_for_cudnn/CMakeLists.txt"),
)
CUDA_MAIN_FILE = Path("dlib/CMakeLists.txt")


class InstallError(RuntimeError):
    """Custom error for installation failures."""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download, patch, and install dlib with CUDA support using the local "
            "GPU compute capability."
        )
    )
    parser.add_argument(
        "--compute-capability",
        dest="compute_capability",
        help=(
            "GPU compute capability in smXY form (e.g., 86). If omitted, the script "
            "attempts to auto-detect using nvidia-smi or PyTorch."
        ),
    )
    parser.add_argument(
        "--dlib-version",
        default=DEFAULT_DLIB_VERSION,
        help="dlib version to install (default: %(default)s)",
    )
    parser.add_argument(
        "pip_args",
        nargs=argparse.REMAINDER,
        help=(
            "Additional arguments forwarded to pip after '--'. Example: "
            "-- --force-reinstall"
        ),
    )
    return parser.parse_args()


def run_command(args: Iterable[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a subprocess command and raise informative errors on failure."""

    result = subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InstallError(
            "Command failed with exit code "
            f"{result.returncode}: {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def normalize_compute_capability(raw_value: str) -> str:
    """Normalize compute capability to a two-digit string (smXY)."""

    cleaned = raw_value.strip().lower().replace("sm", "")
    match = re.fullmatch(r"(\d)(?:[.](\d)|([0-9]))", cleaned)
    if not match:
        raise InstallError(
            "Unable to parse compute capability value '"
            f"{raw_value}'. Expected formats: '86', '8.6', or 'sm_86'."
        )

    major = match.group(1)
    minor = match.group(2) or match.group(3) or "0"
    return f"{major}{minor}"


def detect_compute_capability(explicit: str | None) -> str:
    """Determine the GPU compute capability string."""

    if explicit:
        return normalize_compute_capability(explicit)

    env_override = os.getenv("DLIB_CUDA_ARCH") or os.getenv("CUDA_ARCH")
    if env_override:
        return normalize_compute_capability(env_override)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            line = result.stdout.strip().splitlines()[0]
            return normalize_compute_capability(line)
        except (InstallError, subprocess.CalledProcessError, IndexError):
            pass

    try:
        import torch  # type: ignore import

        if torch.cuda.is_available():  # type: ignore[attr-defined]
            major, minor = torch.cuda.get_device_capability()  # type: ignore[attr-defined]
            return f"{major}{minor}"
    except Exception:  # noqa: BLE001 - best-effort fallback
        pass

    raise InstallError(
        "Unable to determine GPU compute capability. Specify it explicitly via "
        "--compute-capability or set DLIB_CUDA_ARCH environment variable."
    )


def patch_file(path: Path, replacements: list[tuple[re.Pattern[str], str]]) -> None:
    """Apply regex replacements to a file, ensuring at least one matches."""

    content = path.read_text(encoding="utf-8")
    original = content
    for pattern, repl in replacements:
        content, count = pattern.subn(repl, content)
        if count == 0:
            continue
    if content == original:
        raise InstallError(f"No replacements applied to {path}. Pattern mismatch detected.")
    path.write_text(content, encoding="utf-8")


def patch_dlib_sources(source_dir: Path, compute_capability: str) -> None:
    """Update dlib CMake files to use the desired compute capability."""

    arch_token = f"sm_{compute_capability}"
    arch_pattern = re.compile(r"-arch=sm_[0-9]+")
    for relative in CUDA_TEST_FILES:
        patch_file(
            source_dir / relative,
            [(arch_pattern, f"-arch={arch_token}")],
        )

    compute_pattern = re.compile(
        r"set\(DLIB_USE_CUDA_COMPUTE_CAPABILITIES\s+[0-9]+", re.IGNORECASE
    )
    patch_file(
        source_dir / CUDA_MAIN_FILE,
        [(compute_pattern, f"set(DLIB_USE_CUDA_COMPUTE_CAPABILITIES {compute_capability}")],
    )


def download_dlib(version: str, download_dir: Path) -> Path:
    """Download the dlib sdist into download_dir and return extraction path."""

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            f"dlib=={version}",
            "--no-binary",
            ":all:",
            "--no-cache-dir",
        ],
        cwd=download_dir,
    )

    archives = list(download_dir.glob("dlib-*.tar.gz"))
    if not archives:
        raise InstallError("Failed to download dlib source archive.")

    archive_path = archives[0]
    extract_dir = download_dir / archive_path.stem
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(download_dir)
    return extract_dir


def install_dlib(source_dir: Path, compute_capability: str, extra_pip_args: list[str] | None) -> None:
    """Run pip install for the patched dlib sources."""

    env = os.environ.copy()
    extra_flags = f"-DDLIB_USE_CUDA=ON -DDLIB_USE_CUDA_COMPUTE_CAPABILITIES={compute_capability}"
    if existing := env.get("CMAKE_ARGS"):
        env["CMAKE_ARGS"] = f"{existing} {extra_flags}"
    else:
        env["CMAKE_ARGS"] = extra_flags

    if "CUDA_TOOLKIT_ROOT_DIR" not in env:
        env["CUDA_TOOLKIT_ROOT_DIR"] = "/usr/local/cuda"

    cudnn_include = Path("/usr/include/x86_64-linux-gnu/cudnn.h")
    if cudnn_include.exists():
        env.setdefault("CUDNN_INCLUDE_DIR", str(cudnn_include.parent))

    cudnn_lib = Path("/usr/lib/x86_64-linux-gnu/libcudnn.so")
    if cudnn_lib.exists():
        env.setdefault("CUDNN_LIBRARY", str(cudnn_lib))

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            str(source_dir),
            *(extra_pip_args or []),
        ],
        env=env,
    )


def verify_installation() -> None:
    """Ensure the installed dlib reports CUDA support."""

    import dlib  # type: ignore import

    if not getattr(dlib, "DLIB_USE_CUDA", False):
        raise InstallError("dlib installed without CUDA support. Verify CUDA and cuDNN availability.")


def main() -> None:
    """Orchestrate the automated installation workflow."""

    args = parse_args()
    compute_capability = detect_compute_capability(args.compute_capability)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        source_dir = download_dlib(args.dlib_version, tmp_dir)
        patch_dlib_sources(source_dir, compute_capability)
        install_dlib(source_dir, compute_capability, args.pip_args or [])

    verify_installation()
    print(  # noqa: T201 - intended user feedback
        "Successfully installed dlib with CUDA support (sm_" f"{compute_capability})."
    )


if __name__ == "__main__":
    try:
        main()
    except InstallError as error:
        print(f"Error: {error}", file=sys.stderr)  # noqa: T201 - user facing message
        sys.exit(1)

