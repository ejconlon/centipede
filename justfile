python := "env PYTHON_GIL=0 .venv/bin/python3"
packages := "spiny bad_actor minipat pushpluck"
mypy_packages := "-p spiny -p bad_actor -p minipat -p pushpluck"

# No default tasks
default:
  just --list

# Freeze to pip requirements
freeze:
  uv pip freeze > dev-requirements.txt

# Create the virtual environment with pip
venv-frozen:
  python3.13 -m venv --upgrade-deps .venv
  {{python}} -m pip install -r dev-requirements.txt

# Create the virtual environment with uv
venv:
  uv sync

# Format - sort with isort and format with ruff
format:
  {{python}} -m isort --settings-path=pyproject.toml {{packages}} tests
  {{python}} -m ruff format {{packages}} tests

# Typecheck with mypy
typecheck:
  {{python}} -m mypy --config-file=pyproject.toml {{mypy_packages}} -p tests

# Typecheck with mypy - strict mode
typecheck-strict:
  {{python}} -m mypy --strict --cache-dir=.mypy_cache_strict --config-file=pyproject.toml {{mypy_packages}} -p tests

# Lint with ruff
lint:
  {{python}} -m ruff check {{packages}} tests

# Lint with ruff and apply fixes
lint-fix:
  {{python}} -m ruff check --fix {{packages}} tests

# Unit test with pytest in parallel
unit:
  {{python}} -m pytest tests -n auto

# Unit test with pytest (single-threaded)
unit-single-threaded:
  {{python}} -m pytest tests

# Run all tests
test: typecheck unit

# Run all checks
check: format lint test

# Clean most generated files (+ venv)
clean:
  rm -rf .venv .mypy_cache .mypy_cache_strict .pytest_cache *.egg-info

# Enter an Python REPL
repl:
  {{python}}

# Generate HTML documentation
docs:
  rm -rf docs
  {{python}} -m pdoc -o docs -d markdown --include-undocumented {{packages}}

# Start fluidsynth as midi target
fluid:
  fluidsynth -qsi
