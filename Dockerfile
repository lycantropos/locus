ARG IMAGE_NAME
ARG IMAGE_VERSION

FROM ${IMAGE_NAME}:${IMAGE_VERSION}

WORKDIR /opt/locus

COPY pyproject.toml .
COPY README.md .
COPY setup.py .
COPY locus locus
COPY tests tests

RUN pip install -e .[tests]
