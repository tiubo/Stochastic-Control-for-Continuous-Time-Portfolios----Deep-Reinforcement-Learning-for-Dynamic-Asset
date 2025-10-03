# Deep RL Portfolio Allocation - Docker Container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/processed logs

# Expose ports
EXPOSE 8000 8501

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (FastAPI)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative: Run Streamlit dashboard
# CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
