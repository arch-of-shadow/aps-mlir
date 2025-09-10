# Makefile for CADL Frontend (using pixi)

.PHONY: install shell test test-basic test-enhanced test-literals parse lint format clean help

help:
	@echo "Available commands:"
	@echo "  install       Install dependencies using pixi"
	@echo "  shell         Activate pixi shell environment"
	@echo "  test          Run all tests"
	@echo "  test-basic    Run basic parser tests"
	@echo "  test-enhanced Run comprehensive zyy examples tests"
	@echo "  test-literals Run literal width parsing tests"
	@echo "  parse FILE    Parse a CADL file and show summary"
	@echo "  lint          Run linting (mypy, etc.)"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts and cache"

install:
	pixi install

shell:
	pixi shell

test:
	pixi run pytest tests/ -v

test-basic:
	pixi run pytest tests/test_parser.py -v

test-enhanced:
	pixi run pytest tests/test_zyy_examples.py -v

test-literals:
	pixi run pytest tests/test_literal_widths.py -v

parse:
	@if [ -z "$(FILE)" ]; then echo "Usage: make parse FILE=path/to/file.cadl"; exit 1; fi
	pixi run parse $(FILE)

lint:
	pixi run mypy cadl_frontend/ || echo "mypy not configured yet"
	pixi run isort --check-only cadl_frontend/ tests/ || echo "isort not available"
	pixi run black --check cadl_frontend/ tests/ || echo "black not available"

format:
	pixi run isort cadl_frontend/ tests/ || echo "isort not available"
	pixi run black cadl_frontend/ tests/ || echo "black not available"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +