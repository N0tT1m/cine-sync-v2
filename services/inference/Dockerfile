FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY services/inference/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY services/inference /app/services/inference
COPY src /app/src
COPY models /app/models
COPY data /app/data

ENV PYTHONPATH=/app
EXPOSE 8900

CMD ["uvicorn", "services.inference.main:app", "--host", "0.0.0.0", "--port", "8900"]
