BRANCH ?= develop
DOCKER_IMAGE = gwdiexp/lpsd:${BRANCH}
OLD_PY_VERSION = 3.8
NEW_PY_VERSION = 3.11

all: compile mypy test package

compile:
	cd lpsd; \
		gcc -c -fPIC -O3 ltpda_dft.c && \
		gcc -shared -o ltpda_dft.so ltpda_dft.o -Wl,--out-implib,ltpda_dft.a

test:
	PYTHONPATH=`pwd` poetry run py.test

test-watch:
	PYTHONPATH=`pwd` poetry run ptw

test-docker:
	docker run -v `pwd`:/code --rm -it ${DOCKER_IMAGE} make test

mypy:
	poetry run mypy lpsd

pylint:
	poetry run pylint lpsd

black:
	poetry run black lpsd test

package:
	poetry build

upload:
	poetry config pypi-token.pypi ${POETRY_PYPI_TOKEN_PYPI}
	poetry publish

docker:
	docker build . -f docker/Dockerfile -t ${DOCKER_IMAGE} \
		--build-arg PYTHON_VERSION=${NEW_PY_VERSION}
	docker build . -f docker/Dockerfile -t ${DOCKER_IMAGE}-old \
		--build-arg PYTHON_VERSION=${OLD_PY_VERSION}

docker-push:
	docker login
	docker push ${DOCKER_IMAGE}
	docker push ${DOCKER_IMAGE}-old

clean:
	@rm -r dist/

.PHONY: compile mypy test package docker
