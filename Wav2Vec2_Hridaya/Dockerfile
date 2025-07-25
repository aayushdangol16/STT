# Use official Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by torchaudio & ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*  

# Copy requirements and install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app source code and models folder
COPY ./app ./app
COPY ./models ./models

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
