# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create a more robust startup wrapper
RUN echo '#!/bin/bash\n\
echo "Starting Backend API on Port 10001..."\n\
python -m uvicorn api:app --host 0.0.0.0 --port 10001 > /app/api.log 2>&1 & \n\
echo "Starting Streamlit UI on Port $PORT..."\n\
streamlit run app.py --server.port $PORT --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

CMD ["./start.sh"]
