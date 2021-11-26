BRANCH ?= develop
DOCKER_IMAGE = gwdiexp/lpsd:${BRANCH}
OLD_PY_VERSION = 3.7

all: compile mypy test package

compile:
	cd lpsd; \
		gcc -c -fPIC ltpda_dft.c && \
		gcc -shared -o ltpda_dft.so ltpda_dft.o -Wl,--out-implib,ltpda_dft.a

test:
	PYTHONPATH=`pwd` poetry run py.test

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
	docker build . -f docker/Dockerfile -t ${DOCKER_IMAGE}
	docker build . -f docker/Dockerfile -t ${DOCKER_IMAGE}-${OLD_PY_VERSION} \
		--build-arg PYTHON_VERSION=${OLD_PY_VERSION}

docker-push:
	docker login
	docker push ${DOCKER_IMAGE}
	docker push ${DOCKER_IMAGE}-${OLD_PY_VERSION}

clean:
	@rm -r dist/

.PHONY: compile mypy test package docker
