# misfit-agent Makefile

PYTHON ?= python
AGENT_NAME ?= misfit
GAME ?= locksmith

.PHONY: help setup play-local test lint typecheck submit clean

help:
	@echo "Targets:"
	@echo "  setup      install deps via uv"
	@echo "  play-local play one game against the local engine"
	@echo "  test       run pytest"
	@echo "  lint       ruff check"
	@echo "  typecheck  mypy"
	@echo "  submit     build + push Kaggle notebook"
	@echo "  clean      remove build artifacts"

setup:
	uv sync

play-local:
	uv run main.py --agent=$(AGENT_NAME) --game=$(GAME)

test:
	uv run pytest -q

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

submit:
	$(PYTHON) scripts/build_notebook.py
	# kaggle kernels push -p ./kaggle_submission

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
