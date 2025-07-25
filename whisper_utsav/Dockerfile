FROM python:3.10-slim

WORKDIR /app

# Install dependencies (note: install ffmpeg for Whisper to work)
RUN apt-get update && apt-get install -y ffmpeg git && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    jinja2 \
    git+https://github.com/openai/whisper.git

COPY app/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
