# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies required by reportlab fonts
RUN apt-get update \
    && apt-get install -y --no-install-recommends libfreetype6 libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure runtime directories exist
RUN mkdir -p /app/reports

EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} flask_app:app
