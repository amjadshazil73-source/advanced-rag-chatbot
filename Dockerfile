# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for production reliability
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
# We use the CPU-only version of torch to stay within Render's memory and disk limits
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Post-install: Warm up models to bake them into the image
RUN python warmup.py

# Copy the entire project
# This includes 'qdrant_storage' for our baked-in showcase data
COPY . .

# Expose the API and UI ports
EXPOSE 8000
EXPOSE 8501

# Create a robust startup script
# 1. Starts FastAPI on 8000 (Internal)
# 2. Starts Streamlit on $PORT (Public - provided by Render)
RUN echo '#!/bin/bash\n\
uvicorn api:app --host 127.0.0.1 --port 8000 &\n\
streamlit run app.py --server.port $PORT --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Start the application
CMD ["./start.sh"]
