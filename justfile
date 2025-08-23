python := ".venv/bin/python3 -Xgil=0"

# No default tasks
default:
  just --list

# Freeze to pip requirements
freeze:
  uv pip freeze > dev-requirements.txt

# Create the virtual environment
venv:
  python3.13 -m venv --upgrade-deps .venv
  {{python}} -m pip install -r dev-requirements.txt

# Format - sort with isort and format with ruff
format:
  {{python}} -m isort --settings-path=pyproject.toml centipede tests
  {{python}} -m ruff format

# Typecheck with mypy
typecheck:
  {{python}} -m mypy --config-file=pyproject.toml -p centipede
  {{python}} -m mypy --config-file=pyproject.toml -p tests

# Lint with ruff
lint:
  {{python}} -m ruff check

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
  rm -rf .venv .mypy_cache .pytest_cache *.egg-info

# Enter an Python REPL
repl:
  {{python}}

# Run the main entrypoint
main ARGS="":
  {{python}} -m centipede.main {{ARGS}}

# Generate HTML documentation
docs:
  rm -rf docs
  {{python}} -m pdoc -o docs -d markdown --include-undocumented centipede

# Start fluidsynth as midi target
fluid:
  fluidsynth -qsi
