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

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the API and UI ports
EXPOSE 10001
EXPOSE 8501

# Create a robust startup script
# Using 127.0.0.1 for the background API on port 10001
RUN echo '#!/bin/bash\n\
uvicorn api:app --host 127.0.0.1 --port 10001 & \n\
streamlit run app.py --server.port $PORT --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Start the application
CMD ["./start.sh"]
