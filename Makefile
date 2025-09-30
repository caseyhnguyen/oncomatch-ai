.PHONY: install lint format test demo clean help

# Default target
help:
	@echo "OncoMatch AI Development Commands"
	@echo "================================="
	@echo "make install    - Install dependencies"
	@echo "make lint       - Run linter (ruff check)"
	@echo "make format     - Auto-format code"
	@echo "make test       - Run tests"
	@echo "make demo       - Run demo with Kerry Bird patient"
	@echo "make clean      - Remove cache and temp files"
	@echo "make help       - Show this help message"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

lint:
	ruff check src tests

format:
	ruff check --select I --fix src tests
	ruff format src tests

test:
	pytest -q

demo:
	@echo "Running OncoMatch AI Demo..."
	@PYTHONPATH=src python -m oncomatch.match --patient_id "Kerry Bird" --format human

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info
