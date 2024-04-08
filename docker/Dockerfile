ARG PYTHON_VERSION=3

FROM python:$PYTHON_VERSION

ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code

ENV BUILD_PACKAGES build-essential python3-distutils

RUN apt-get update -y && apt-get install -y $BUILD_PACKAGES

RUN pip install --no-cache --upgrade poetry setuptools

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /code/
RUN poetry install -vvv --no-interaction
