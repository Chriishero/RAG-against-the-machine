PYTHON = python3
UV = uv
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
				--disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync --active

run:
	$(UV) run --active $(PYTHON) -m src

debug:
	$(UV) run --active $(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	$(UV) run $(PYTHON) -m flake8 .
	$(UV) run $(PYTHON) -m mypy $(MYPY_FLAGS) .

lint-strict:
	$(UV) run $(PYTHON) -m flake8.
	$(UV) run $(PYTHON) -m mypy --strict .
