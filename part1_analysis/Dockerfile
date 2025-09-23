# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for matplotlib and optional PDF generation
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    pkg-config \
    # Optional: uncomment these lines if you need PDF generation
    # pandoc \
    # texlive-xetex \
    # texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the analysis script
COPY analysis_new.py .

# Create directories for input, output, and data
RUN mkdir -p /app/data /app/output /app/images

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set default command
ENTRYPOINT ["python", "analysis_new.py"]

# Default arguments (can be overridden)
CMD ["--help"]