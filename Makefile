PYTHON ?= python

all: compile test package

compile:
	$(PYTHON) build.py
	$(PYTHON) -m lpsd_fast.build

test:
	$(PYTHON) -m pytest

test-watch:
	$(PYTHON) -m pytest_watch

mypy:
	$(PYTHON) -m mypy lpsd

pylint:
	$(PYTHON) -m pylint lpsd

black:
	$(PYTHON) -m black lpsd test

package:
	$(PYTHON) -m build

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist')]"

.PHONY: all compile test test-watch mypy pylint black package clean
