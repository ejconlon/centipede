# No default tasks
default:
  just --list

# Create the virtual environment
venv:
  python3.13 -m venv --upgrade-deps .venv
  .venv/bin/python3 -m pip install wheel -r dev-requirements.txt -r requirements.txt

# Format - for now just sort imports
format:
  .venv/bin/python3 -m isort --settings-path=pyproject.toml centipede tests
  .venv/bin/python3 -m blue --config=pyproject.toml centipede tests

# Typecheck with mypy
typecheck:
  .venv/bin/python3 -m mypy --config-file=pyproject.toml -p centipede
  .venv/bin/python3 -m mypy --config-file=pyproject.toml -p tests

# Lint with flake8
lint:
  .venv/bin/python3 -m flake8 --config=pyproject.toml centipede tests

# Unit test with pytest
unit:
  .venv/bin/python3 -m pytest tests

# Run all tests
test: lint typecheck unit

# Clean most generated files (+ venv)
clean:
  rm -rf .venv .mypy_cache .pytest_cache *.egg-info

# Enter an IPython REPL
repl:
  .venv/bin/ipython

# Run the main entrypoint
exe:
  .venv/bin/python3 -m centipede.main
