# Оfficial Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Astral's uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile

# Set the working directory
WORKDIR /code

# Copy dependency manifests first for caching
COPY pyproject.toml uv.lock pipeline.bin ./

# Install project dependencies (locked, no dev packages)
RUN /root/.local/bin/uv sync --frozen --no-dev

# Copy FastAPI app
COPY predict.py ./

# Expose API port
EXPOSE 9696

# Start the FastAPI server
CMD ["/root/.local/bin/uv", "run", "--no-sync", "uvicorn", "predict:app", "--host=0.0.0.0", "--port=9696"]
