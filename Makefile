PYTHON ?= python
CC ?= cc
export CC

all: compile test package

compile:
	$(PYTHON) build_native.py
	$(PYTHON) -m lpsd_fast.build

compile-native:
	$(PYTHON) build_native.py
	$(PYTHON) -m lpsd_fast.build --native

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

.PHONY: all compile compile-native test test-watch mypy pylint black package clean
