# On Windows use Git Bash, or run the commands below by hand.
PY ?= python

.PHONY: install dev test lint typecheck check collection index eval

install:            ## create the venv and install with dev tools
	$(PY) -m venv .venv && .venv/bin/pip install -e ".[dev]" || .venv/Scripts/pip install -e ".[dev]"

dev:                ## run with reload
	uvicorn app.main:app --reload --app-dir src --port 8030

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

typecheck:
	mypy

check: lint typecheck test   ## what CI runs

collection:         ## download the photographs from The Met's Open Access API
	PYTHONPATH=src python scripts/fetch_collection.py

index:              ## embed every photograph (GPU, about 2.5 GB)
	PYTHONPATH=src python scripts/build_index.py

eval:               ## measure picture, caption and hybrid search against The Met's subject tags
	PYTHONPATH=src python scripts/eval_search.py
