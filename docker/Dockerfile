# Use official slim Python runtime as base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV, MediaPipe, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements list first to leverage Docker cache
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, and models
COPY src/ /app/src/
COPY models/ /app/models/
COPY configs/ /app/configs/

# Expose port for Streamlit dashboard
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
