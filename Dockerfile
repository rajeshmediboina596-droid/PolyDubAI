FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system dependencies: FFmpeg, OpenCV runtime libraries, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . /app/

# Ensure output, models, and temp directories exist with write permissions
RUN mkdir -p /app/output /app/models /app/web && chmod -R 777 /app/output /app/models

# Expose port (7860 is default for Hugging Face Spaces; Render/Railway pass $PORT)
EXPOSE 7860

# Run FastAPI server
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
