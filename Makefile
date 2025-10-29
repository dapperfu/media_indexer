.PHONY: all install install-dev test lint doc-check format format-check check validate-requirements generate-sdocs-html clean help

# Python and virtual environment setup
VENV_DIR = venv
VENV_PYTHON = ${VENV_DIR}/bin/python
VENV_PIP = ${VENV_DIR}/bin/pip
VENV_UV = ${VENV_DIR}/bin/uv
PYTHON = python3
PYTEST = ${VENV_DIR}/bin/pytest
RUFF = ${VENV_DIR}/bin/ruff
STRICTDOC = ${VENV_DIR}/bin/strictdoc
PYDOCSTYLE = ${VENV_DIR}/bin/pydocstyle

# Default target - venv installs everything
all: venv

# Create virtual environment and install all dependencies
venv: ${VENV_PYTHON}

${VENV_PYTHON}:
	${PYTHON} -m venv ${VENV_DIR}
	${VENV_PIP} install uv
	${VENV_UV} pip install -e ".[dev]"

# Install dependencies (reinstall/upgrade)
install: ${VENV_UV}
	${VENV_UV} pip install -e .

# Install development dependencies
install-dev: ${VENV_UV}
	${VENV_UV} pip install -e ".[dev]"

# Run tests
test: ${PYTEST}
	${PYTEST} tests/

# Run linters
lint: ${RUFF}
	${RUFF} check src/

# Check formatting (does not format)
format-check: ${RUFF}
	${RUFF} format --check src/

# Format code
format: ${RUFF}
	${RUFF} format src/
	${RUFF} check --fix src/

# Run all checks
check: lint doc-check test

# Check documentation style
doc-check: ${PYDOCSTYLE}
	${PYDOCSTYLE} src/media_indexer --convention=numpy

# Validate requirements.sdoc
validate-requirements: ${STRICTDOC}
	${STRICTDOC} export requirements.sdoc --output-dir /tmp/strictdoc_export_validate

# Generate HTML from all .sdoc files
generate-sdocs-html: ${STRICTDOC}
	mkdir -p docs
	${STRICTDOC} export requirements.sdoc --output-dir docs/requirements

# Clean generated files
clean:
	rm -rf ${VENV_DIR}
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf .checkpoint.json
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf docs/

# Help
help:
	@echo "Available targets:"
	@echo "  make venv              - Create virtual environment"
	@echo "  make install           - Install dependencies"
	@echo "  make install-dev       - Install development dependencies"
	@echo "  make test              - Run tests"
	@echo "  make lint              - Run ruff linter"
	@echo "  make doc-check         - Check documentation style (pydocstyle)"
	@echo "  make format            - Format code with ruff"
	@echo "  make format-check      - Check formatting without fixing"
	@echo "  make check             - Run all checks (lint + doc-check + test)"
	@echo "  make validate-requirements - Validate requirements.sdoc"
	@echo "  make generate-sdocs-html - Generate HTML from .sdoc files"
	@echo "  make clean             - Clean generated files"

