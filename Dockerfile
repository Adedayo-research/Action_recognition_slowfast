# Use the official NVIDIA PyTorch image as the base
# This image already contains CUDA 11.8 and cuDNN baked into the OS
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OpenCV and video processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the exact Python dependencies we generated earlier
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire experiment_2 codebase into the container
COPY experiment_2 /app/experiment_2

# Define the default command to run when the container starts
CMD ["python", "/app/experiment_2/scripts/training_unfrozen_blocks4_5.py"]
