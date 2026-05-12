# Use a slim Python 3.10 image to keep the container lightweight
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# IMPORTANT: We explicitly install the CPU version of PyTorch first.
# This prevents pip from downloading the 5GB+ CUDA version which would cause
# Railway deployments to fail or be extremely slow.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY . .

# Expose the port (Railway automatically injects the PORT env variable)
EXPOSE $PORT

# Run Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
