# Use Python 3.11 as the base image
FROM python:3.11-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including texlive-full
RUN apt update && apt install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

COPY . /app/
RUN uv sync

# Set the default command to an empty array
CMD ["/bin/bash"]