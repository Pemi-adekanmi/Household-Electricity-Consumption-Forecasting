# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY train.py .
COPY predict.py .

# Create directories for data and models
RUN mkdir -p data models

# Expose API port
EXPOSE 8000

# Default command: run the API
CMD ["python", "predict.py"]

