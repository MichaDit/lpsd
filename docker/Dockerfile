ARG PYTHON_VERSION=3.10

FROM python:$PYTHON_VERSION

ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code

ENV BUILD_PACKAGES build-essential

RUN apt-get update -y && apt-get install -y $BUILD_PACKAGES

RUN pip install --no-cache poetry

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /code/
RUN poetry install -vvv --no-interaction
