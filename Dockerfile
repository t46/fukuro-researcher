FROM python:3.11-slim

# Install necessary system tools and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /root

# Copy local files to the container
COPY . /root/

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN pip install --no-cache-dir --upgrade pip && pip install uv

RUN uv sync

# Grant execution permission to the script
RUN chmod +x /root/entrypoint.sh

# Set the default command to bash when the container starts
CMD ["/root/entrypoint.sh"]