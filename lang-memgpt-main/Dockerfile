# syntax=docker/dockerfile:1.4
FROM node:alpine AS frontend-build

WORKDIR /app/frontend
COPY new-alfred-ui/package*.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm ci --production

# Copy the entire src directory first
COPY new-alfred-ui/src ./src
COPY new-alfred-ui/public ./public

# Add logging to debug the build process
RUN echo "Contents of /app/frontend:" && ls -la && \
    npm run build && \
    echo "Contents of /app/frontend/build:" && ls -la build


# Use slim Python image and combine installation steps
FROM python:3.11-slim AS python-base

# Install system dependencies in a single layer
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        portaudio19-dev \
        python3-pyaudio && \
    rm -rf /var/lib/apt/lists/*

# Set Poetry environment variables
ENV POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VENV="/opt/poetry-venv" \
    POETRY_CACHE_DIR="/opt/.cache" \
    PYTHONUNBUFFERED=1

# Install Poetry with pip cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv $POETRY_VENV && \
    $POETRY_VENV/bin/pip install -U pip setuptools && \
    $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

# Copy dependency files
COPY poetry.lock pyproject.toml ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction

# Final minimal stage
FROM python:3.11-slim

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        portaudio19-dev \
        python3-pyaudio && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir poetry uvicorn fastapi

# Copy dependency files
COPY poetry.lock pyproject.toml ./

# Install project dependencies (updated flags)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction

# Copy Python environment and frontend build
COPY --from=frontend-build /app/frontend/build/static /app/static/static
COPY --from=frontend-build /app/frontend/build/* /app/static/

# Copy application code
COPY . /app/

WORKDIR /app

# Expose port and set command
EXPOSE $PORT

# Use shell form to allow environment variable expansion
CMD uvicorn api:app --host 0.0.0.0 --port=${PORT:-8000}
