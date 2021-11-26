compile:
	cd lpsd; \
		gcc -c -fPIC ltpda_dft.c && \
		gcc -shared -o ltpda_dft.so ltpda_dft.o -Wl,--out-implib,ltpda_dft.a

test:
	poetry run py.test -x

mypy:
	poetry run mypy lpsd

pylint:
	poetry run pylint lpsd

black:
	poetry run black

package:
	poetry build

upload:
	poetry config pypi-token.pypi ${POETRY_PYPI_TOKEN_PYPI}
	poetry publish

clean:
	@rm -r dist/ build/

.PHONY: compile mypy test package
