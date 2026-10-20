.PHONY: install test lint format
install:
	uv sync --all-extras
test:
	pytest -q
lint:
	ruff check . && ruff format --check .
format:
	ruff check --fix . && ruff format .
