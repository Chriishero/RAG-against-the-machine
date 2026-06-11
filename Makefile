PYTHON = python3
UV = uv
CMD ?=
ARGS ?=
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
				--disallow-untyped-defs --check-untyped-defs

ifdef VIRTUAL_ENV
	ACTIVE = --active
else
	ACTIVE =
endif

all: install

install:
	$(UV) sync $(ACTIVE)
	ollama pull qwen3:0.6b

run:
ifeq ($(CMD),)
	$(error you must use one of these command: index, search, answer, search_dataset, answer_dataset. Ex: make run CMD=index)
endif
	$(UV) run $(ACTIVE) $(PYTHON) -m src $(CMD) $(ARGS)

debug:
ifeq ($(CMD),)
	$(error you must use one of these command: index, search, answer, search_dataset, answer_dataset. Ex: make run CMD=index)
endif
	$(UV) run $(ACTIVE) $(PYTHON) -m pdb -m src $(CMD) $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .venv dist .cache/ .pytest_cache
	rm -rf data/output/ data/processed/

lint:
	$(UV) run $(ACTIVE) $(PYTHON) -m flake8 src/
	$(UV) run $(ACTIVE) $(PYTHON) -m mypy $(MYPY_FLAGS) src/

lint-strict:
	$(UV) run $(ACTIVE) $(PYTHON) -m flake8 src/
	$(UV) run $(ACTIVE) $(PYTHON) -m mypy --strict src/

help:
	@echo "Usage:"
	@echo "	make install"
	@echo "	make run CMD=<command> ARGS='<arguments>'"
	@echo "	make debug CMD=<command> ARGS='<arguments>'"
	@echo ""
	@echo "Examples:"
	@echo "	make run CMD=index"
	@echo "	make run CMD=search ARGS='\"What is Retrieval Augmented Generation?\" --k 10'"
	@echo "	make run CMD='search \"What is Retrieval Augmented Generation?\" --k 10'"

.PHONY: all install run debug clean lint lint-strict help