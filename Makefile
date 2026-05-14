.PHONY: env env-update lock install test test-fast lint format typecheck check clean

# One-time environment creation
env:
	conda env create -f environment.yml
	@echo "Now run: conda activate galaxy-morph && make install"

# Update environment when environment.yml changes
env-update:
	conda env update -f environment.yml --prune

# Regenerate the lock file
lock:
	conda-lock -f environment.yml -p osx-arm64 -p linux-64

# Install the package itself (assumes env is activated)
install:
	pip install -e .
	pre-commit install

test-fast:
	pytest -m "not slow and not gpu and not integration" -n auto

test:
	pytest -m "not gpu" -n auto

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

check: lint typecheck test-fast

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
