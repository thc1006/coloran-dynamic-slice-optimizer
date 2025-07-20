FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     python3.9     python3-pip     git     build-essential     && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.coloran_optimizer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
