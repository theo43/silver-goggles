# Use official Python 3.10 image
FROM python:3.10-slim

# Set environment variables to avoid interactive prompts and optimize Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for image processing, curl, and MongoDB interaction
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Download the ImageNet labels JSON file using curl
RUN curl -o imagenet_labels.json https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/imagenet1000_clsidx_to_labels.txt

# Copy the requirements.txt and install the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port for FastAPI
EXPOSE 8082

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
