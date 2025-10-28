.PHONY: all venv install install-dev test lint lint-format format check validate-requirements clean help

# Python and virtual environment setup
VENV_DIR = venv
PYTHON = python3
PIP = ${VENV_DIR}/bin/pip
PYTEST = ${VENV_DIR}/bin/pytest
RUFF = ${VENV_DIR}/bin/ruff

# Default target
all: venv install

# Create virtual environment
venv:
	${PYTHON} -m venv ${VENV_DIR}

# Install dependencies
install: venv
	${PIP} install -e .

# Install development dependencies
install-dev: venv
	${PIP} install -e ".[dev]"

# Run tests
test: venv
	${PYTEST} tests/

# Run linters
lint: venv
	${RUFF} check src/

# Check formatting
lint-format: venv
	${RUFF} format --check src/

# Format code
format: venv
	${RUFF} format src/
	${RUFF} check --fix src/

# Run all checks
check: lint test

# Validate requirements.sdoc
validate-requirements:
	strictdoc export requirements.sdoc --output-dir /tmp/strictdoc_export

# Clean generated files
clean:
	rm -rf ${VENV_DIR}
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf .checkpoint.json
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

# Help
help:
	@echo "Available targets:"
	@echo "  make venv              - Create virtual environment"
	@echo "  make install           - Install dependencies"
	@echo "  make install-dev       - Install development dependencies"
	@echo "  make test              - Run tests"
	@echo "  make lint              - Run ruff linter"
	@echo "  make lint-format       - Check formatting without fixing"
	@echo "  make format            - Format code with ruff"
	@echo "  make check             - Run all checks (lint + test)"
	@echo "  make validate-requirements - Validate requirements.sdoc"
	@echo "  make clean             - Clean generated files"

