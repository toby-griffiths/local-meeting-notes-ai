FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY index.html ./index.html

# Unprivileged runtime user. /data holds meeting history and temp audio and
# is the only writable app location (mounted as a volume by docker-compose).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data \
    && chmod 700 /data

ENV DATA_DIR=/data
WORKDIR /data
USER app

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "/app/app", "main:app", "--host", "0.0.0.0", "--port", "8000"]
