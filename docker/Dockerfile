ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}
ENV PYTHONUNBUFFERED=1
WORKDIR /code
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY . /code
RUN python -m pip install --no-cache-dir '.[test,dev]'
CMD ["make", "test"]
