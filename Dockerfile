# Using an official PyTorch image with CUDA support and Python 3.10
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (essential for OpenCV and image processing libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project from your local folder into /app inside the container
COPY . .

# Create input and output directories inside the container
RUN mkdir -p input output

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:/app/models"

# Correct entrypoint pointing to the models subfolder
ENTRYPOINT ["python", "models/menual.py"]