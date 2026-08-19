MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

lint:
	uv run flake8 src 
	uv run mypy $(MYPY_FLAGS) src 

lint-strict:
	uv run mypy --strict --ignore-missing-imports src 

test:
	uv run pytest
